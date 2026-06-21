#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import math
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from gaussian_renderer import render, drk_render_func

MCMC_N_MAX = 51
_MCMC_BINOMS = None


def get_mcmc_binoms(device):
    global _MCMC_BINOMS
    if _MCMC_BINOMS is None or _MCMC_BINOMS.device != device:
        binoms = torch.zeros((MCMC_N_MAX, MCMC_N_MAX), dtype=torch.float32, device=device)
        for n in range(MCMC_N_MAX):
            for k in range(n + 1):
                binoms[n, k] = math.comb(n, k)
        _MCMC_BINOMS = binoms
    return _MCMC_BINOMS


def compute_mcmc_relocation(opacity_old, scale_old, N):
    N = N.to(dtype=torch.long).clamp_(min=1, max=MCMC_N_MAX - 1)
    opacity_old = opacity_old.clamp(
        min=torch.finfo(torch.float32).eps,
        max=1.0 - torch.finfo(torch.float32).eps,
    )
    opacity_new = 1.0 - torch.pow(1.0 - opacity_old, 1.0 / N.to(opacity_old.dtype))

    binoms = get_mcmc_binoms(opacity_old.device)
    denom_sum = torch.zeros_like(opacity_new)
    for i in range(1, MCMC_N_MAX):
        active = N >= i
        if not active.any():
            break
        cur_sum = torch.zeros_like(opacity_new)
        for k in range(i):
            sign = -1.0 if (k % 2) else 1.0
            cur_sum = cur_sum + sign * binoms[i - 1, k] * torch.pow(opacity_new, k + 1) / math.sqrt(k + 1)
        denom_sum = torch.where(active, denom_sum + cur_sum, denom_sum)

    coeff = (opacity_old / denom_sum.clamp_min(torch.finfo(torch.float32).eps)).unsqueeze(-1)
    return opacity_new, coeff * scale_old


class GaussianModel:

    def update(*args, **kwargs):
        return

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def train(self):
        self.training = True
        return self
    
    def eval(self):
        self.training = False
        return self

    @property
    def min_opacity_pruning(self):
        return self._min_opacity_pruning

    def __init__(self, sh_degree : int):
        self.render_func = render
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._min_opacity_pruning = 0.005
        self.densify_grad_threshold = 0.00025
        self.is_2D = False
        self.baking_mode = False
        self.training = False
        self.current_opt_step = 0
        self.cache_sort = False
        self.use_mcmc = False
        self.mcmc_strategy = "replace"
        self.mcmc_start_iter = -1
        self.mcmc_end_iter = -1
        self.mcmc_cap_max = -1
        self.mcmc_growth_rate = 1.05
        self.mcmc_grad_weight = 1.5
        self.mcmc_scale_weight = 0.5
        self.mcmc_min_opacity = 0.005
        self.mcmc_noise_lr = 0.0
        self.mcmc_opacity_reg = 0.0
        self.mcmc_scale_reg = 0.0
        self.mcmc_prune_min_opacity = 0.0
        self.mcmc_prune_score = "opacity"
        self.current_xyz_lr = 0.0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def grad_postprocess(self):
        pass

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.densification_interval = training_args.densification_interval
        self.opacity_reset_interval = training_args.opacity_reset_interval
        self.use_mcmc = getattr(training_args, "use_mcmc", False)
        self.mcmc_strategy = getattr(training_args, "mcmc_strategy", "replace")
        self.mcmc_start_iter = getattr(training_args, "mcmc_start_iter", -1)
        self.mcmc_end_iter = getattr(training_args, "mcmc_end_iter", -1)
        self.mcmc_cap_max = getattr(training_args, "mcmc_cap_max", -1)
        self.mcmc_growth_rate = getattr(training_args, "mcmc_growth_rate", 1.05)
        self.mcmc_grad_weight = getattr(training_args, "mcmc_grad_weight", 0.0)
        self.mcmc_scale_weight = getattr(training_args, "mcmc_scale_weight", 0.0)
        self.mcmc_min_opacity = getattr(training_args, "mcmc_min_opacity", 0.005)
        self.mcmc_noise_lr = getattr(training_args, "mcmc_noise_lr", 0.0)
        self.mcmc_opacity_reg = getattr(training_args, "mcmc_opacity_reg", 0.0)
        self.mcmc_scale_reg = getattr(training_args, "mcmc_scale_reg", 0.0)
        self.mcmc_prune_min_opacity = getattr(training_args, "mcmc_prune_min_opacity", 0.0)
        self.mcmc_prune_score = getattr(training_args, "mcmc_prune_score", "opacity")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                self.current_xyz_lr = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], {})
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                if 'step' not in stored_state:
                    stored_state['step'] = torch.zeros([], dtype=torch.float32).cpu()
                try:
                    del self.optimizer.state[group['params'][0]]
                except:
                    pass
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if not len(group["params"]) == 1:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if not len(group["params"]) == 1:
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def reset_optimizer_state(self, inds=None):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if not len(group["params"]) == 1:
                continue
            tensor = group["params"][0]
            stored_state = self.optimizer.state.get(tensor, None)
            if stored_state is not None:
                if inds is None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                else:
                    stored_state["exp_avg"][inds] = 0
                    stored_state["exp_avg_sq"][inds] = 0
                del self.optimizer.state[tensor]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def refresh_parameters_from_optimizer(self, optimizable_tensors):
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def _sample_mcmc_sources(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs, minlength=self.get_xyz.shape[0]).unsqueeze(-1)
        return sampled_idxs, ratio

    def _mcmc_updated_params(self, idxs, ratio):
        new_opacity, new_scaling = compute_mcmc_relocation(
            opacity_old=self.get_opacity[idxs, 0],
            scale_old=self.get_scaling[idxs],
            N=ratio[idxs, 0] + 1,
        )
        new_opacity = torch.clamp(
            new_opacity.unsqueeze(-1),
            min=self.mcmc_min_opacity,
            max=1.0 - torch.finfo(torch.float32).eps,
        )
        return {
            "xyz": self._xyz[idxs],
            "f_dc": self._features_dc[idxs],
            "f_rest": self._features_rest[idxs],
            "opacity": self.inverse_opacity_activation(new_opacity),
            "scaling": self.scaling_inverse_activation(new_scaling),
            "rotation": self._rotation[idxs],
        }

    def _mcmc_assign_params(self, indices, params):
        self._xyz.data[indices] = params["xyz"]
        self._features_dc.data[indices] = params["f_dc"]
        self._features_rest.data[indices] = params["f_rest"]
        self._opacity.data[indices] = params["opacity"]
        self._scaling.data[indices] = params["scaling"]
        self._rotation.data[indices] = params["rotation"]

    def _mcmc_copy_source_params(self, source_indices, target_indices):
        self._opacity.data[source_indices] = self._opacity.data[target_indices]
        self._scaling.data[source_indices] = self._scaling.data[target_indices]

    def _mcmc_append_params(self, params):
        self.densification_postfix(
            params["xyz"],
            params["f_dc"],
            params["f_rest"],
            params["opacity"],
            params["scaling"],
            params["rotation"],
        )

    @torch.no_grad()
    def _mcmc_source_probs(self, indices=None):
        """Sampling weight for MCMC clone/relocate sources. Default = opacity (3DGS-MCMC).
        With mcmc_grad_weight>0, multiply by a factor of the per-primitive ABSOLUTE view-space
        gradient (xyz_gradient_accum is accumulated from the CUDA abs-grad densify buffer, AbsGS):
        biases new primitives toward UNDER-RECONSTRUCTED regions (e.g. blurry distant areas where
        one large primitive spans high-frequency content) so detail gets sculpted, not just the
        already-opaque near field."""
        op = self.get_opacity.squeeze(-1)
        op = op[indices] if indices is not None else op
        w = float(getattr(self, "mcmc_grad_weight", 0.0))
        if w > 0.0 and self.denom.numel() > 0:
            g = (self.xyz_gradient_accum.squeeze(-1) / self.denom.squeeze(-1).clamp_min(1.0)).nan_to_num(0.0)
            g = g[indices] if indices is not None else g
            gn = g / g.quantile(0.95).clamp_min(1e-12)          # normalize by 95th percentile
            if float(getattr(self, "mcmc_scale_weight", 0.0)) > 0.0:   # also favor large primitives
                s = self.get_scaling.max(dim=1).values
                s = s[indices] if indices is not None else s
                sn = s / s.quantile(0.95).clamp_min(1e-12)
                gn = gn + float(self.mcmc_scale_weight) * sn
            return op * (1.0 + w * gn.clamp(0.0, 10.0))
        return op

    @torch.no_grad()
    def relocate_gs(self, dead_mask):
        if dead_mask.sum() == 0:
            return 0
        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
        if alive_indices.shape[0] == 0:
            return 0

        probs = self._mcmc_source_probs(alive_indices)
        reinit_idx, ratio = self._sample_mcmc_sources(probs, dead_indices.shape[0], alive_indices=alive_indices)
        params = self._mcmc_updated_params(reinit_idx, ratio)
        self._mcmc_assign_params(dead_indices, params)
        self._mcmc_copy_source_params(reinit_idx, dead_indices)
        self.refresh_parameters_from_optimizer(self.reset_optimizer_state(inds=torch.cat([dead_indices, reinit_idx]).unique()))
        return int(dead_indices.shape[0])

    @torch.no_grad()
    def add_new_gs(self, cap_max=None):
        cap_max = self.mcmc_cap_max if cap_max is None else cap_max
        current_num_points = self._opacity.shape[0]
        if cap_max is None or cap_max <= 0:
            target_num = int(self.mcmc_growth_rate * current_num_points)
        else:
            target_num = min(cap_max, int(self.mcmc_growth_rate * current_num_points))
        num_gs = max(0, target_num - current_num_points)
        if num_gs <= 0:
            return 0

        add_idx, ratio = self._sample_mcmc_sources(self._mcmc_source_probs(), num_gs)
        params = self._mcmc_updated_params(add_idx, ratio)
        self._opacity.data[add_idx] = params["opacity"]
        self._scaling.data[add_idx] = params["scaling"]
        self._mcmc_append_params(params)
        self.refresh_parameters_from_optimizer(self.reset_optimizer_state(inds=add_idx))
        return num_gs

    def mcmc_densify(self):
        dead_mask = (self.get_opacity <= self.mcmc_min_opacity).squeeze(-1)
        relocated = self.relocate_gs(dead_mask)
        added = self.add_new_gs()
        pruned = self.mcmc_prune()
        return relocated, added, pruned

    @torch.no_grad()
    def mcmc_prune(self):
        if self.get_xyz.shape[0] == 0:
            return 0
        if self.mcmc_prune_min_opacity > 0.0:
            prune_mask = (self.get_opacity < self.mcmc_prune_min_opacity).squeeze(-1)
        else:
            prune_mask = torch.zeros((self.get_xyz.shape[0],), dtype=torch.bool, device=self.get_xyz.device)
        if self.mcmc_cap_max is not None and self.mcmc_cap_max > 0 and self.get_xyz.shape[0] > self.mcmc_cap_max:
            extra = int(self.get_xyz.shape[0] - self.mcmc_cap_max)
            score = self._mcmc_prune_score()
            _, low_idx = torch.topk(score, extra, largest=False)
            prune_mask[low_idx] = True
        pruned = int(prune_mask.sum().item())
        if pruned > 0 and pruned < self.get_xyz.shape[0]:
            self.prune_points(prune_mask)
        return pruned

    @torch.no_grad()
    def _mcmc_prune_score(self):
        opacity = self.get_opacity.squeeze(-1)
        if getattr(self, "mcmc_prune_score", "opacity") == "contrib":
            visible = self.denom.squeeze(-1).clamp_min(1.0)
            grad = (self.xyz_gradient_accum.squeeze(-1) / visible).nan_to_num(0.0)
            grad_norm = grad / grad.detach().quantile(0.95).clamp_min(1e-8)
            visibility = (visible / visible.detach().max().clamp_min(1.0)).clamp(0.05, 1.0)
            return opacity * visibility * (0.25 + grad_norm.clamp(0.0, 4.0))
        return opacity

    @torch.no_grad()
    def add_mcmc_noise(self):
        if self.mcmc_noise_lr <= 0.0 or self.current_xyz_lr <= 0.0 or self.get_xyz.shape[0] == 0:
            return
        L = build_scaling_rotation(self.get_scaling, self.get_rotation)
        actual_covariance = L @ L.transpose(1, 2)

        def op_sigmoid(x, k=100, x0=0.995):
            return 1 / (1 + torch.exp(-k * (x - x0)))

        noise = torch.randn_like(self._xyz) * op_sigmoid(1 - self.get_opacity) * self.mcmc_noise_lr * self.current_xyz_lr
        noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
        self._xyz.add_(noise)

    def regularization_loss(self, **kwargs):
        loss = 0.0
        if self.mcmc_opacity_reg > 0.0:
            loss = loss + self.mcmc_opacity_reg * torch.abs(self.get_opacity).mean()
        if self.mcmc_scale_reg > 0.0:
            loss = loss + self.mcmc_scale_reg * torch.abs(self.get_scaling).mean()
        return loss

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, **kwargs):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def create_from_gs2d(self, gs2d, sample_rate=1.):
        if sample_rate < 1.:
            gs_num = int(gs2d.get_xyz.shape[0] * sample_rate)
            samp_idx = farthest_point_sample(gs2d.get_xyz[None], gs_num)[0]
        else:
            samp_idx = torch.ones_like(gs2d._xyz[..., 0], dtype=bool)
        self._features_dc = nn.Parameter(gs2d._features_dc[samp_idx])
        self._features_rest = nn.Parameter(gs2d._features_rest[samp_idx])
        self._xyz = nn.Parameter(gs2d._xyz[samp_idx])
        self._opacity = nn.Parameter(gs2d._opacity[samp_idx])
        self._rotation = nn.Parameter(gs2d._rotation[samp_idx])
        self._scaling = nn.Parameter(self.scaling_inverse_activation(gs2d.get_scaling[..., :2]).mean(dim=-1, keepdim=True).expand(gs2d._scaling.shape[0], 3).clone()[samp_idx])
        self.max_radii2D = torch.zeros_like(self._xyz[..., 0])
        self.baking_mode = True

    def create_from_gs2dgs(self, gs2dgs, sample_rate=1.):
        self.create_from_gs2d(gs2d=gs2dgs, sample_rate=sample_rate)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def get_range_activation(min=0., max=1.):
    def activation(x):
        x = torch.sigmoid(x) * (max - min) + min
        return x
    return activation


def get_range_inv_activation(min=0., max=1.):
    def inv_activation(x):
        x = inverse_sigmoid((x - min) / (max - min))
        return x
    return inv_activation


@torch.no_grad()
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


class DRKModel(GaussianModel):
    def __init__(self, sh_degree: int, kernel_K=8, max_theta_rate=5., sharpen_alpha=False, training=False):
        super().__init__(sh_degree)
        self.kernel_K = kernel_K
        self._acutance = torch.empty(0)
        self._thetas = torch.empty(0)
        self._l1l2_rates = torch.empty(0)
        self.render_func = drk_render_func
        self.sharpen_alpha = sharpen_alpha

        self.is_2D = True
        self.final_min_opacity_pruning = 1e-2
        self.densify_grad_threshold = 1e-3
        self.opacity_densify_grad_threshold = 5e-5
        self.training = training
        
        self.key_stages                  = [500,        10000,      15000,      25000]
        self.acutance_interval_list      = [[-.1, .50], [-.1, .75], [-.1, .85], [-.1, .99]]
        self.reset_opacity_interval_list = [3000,       3000,       3000,       int(1e7)]
        self.densification_interval_list = [200,        200,        200,        int(1e7)]
        self.scales_freedom_list         = [8,          8,          8,          8]
        self.l1l2rates_free_stage        = -2
        self.current_stage               = -1
        
        self.scales_freedom              = 8
        self.acutance_min, self.acutance_max = -.001, .001
        self.acutance_activation = get_range_activation(self.acutance_min, self.acutance_max)
        self.inv_acutance_activation = get_range_inv_activation(self.acutance_min, self.acutance_max)

        self.max_theta_rate = min(max_theta_rate, self.kernel_K-1)  # Avoid the angle being over pi/2
        self.theta_residual = 0  # 1 / (max_theta_rate - 1 + 1e-5)
        self.clone_only = False
        self.no_recenter = False
        self.no_resetopacity = False

        self.cache_sort = False
        self.tile_culling = False
        self.use_mcmc = False
        self.mcmc_strategy = "replace"
        self.mcmc_start_iter = -1
        self.mcmc_end_iter = -1
        self.mcmc_cap_max = -1
        self.mcmc_growth_rate = 1.05
        self.mcmc_grad_weight = 1.5
        self.mcmc_scale_weight = 0.5
        self.mcmc_min_opacity = 0.005
        self.mcmc_noise_lr = 0.0
        self.mcmc_opacity_reg = 0.0
        self.mcmc_scale_reg = 0.0
        self.mcmc_prune_min_opacity = 0.0
        self.mcmc_prune_score = "opacity"

    def train(self):
        self.training = True
        return self
    
    def eval(self):
        self.training = False
        return self
    
    def update(self, iteration):
        self.iteration = iteration
        if iteration in self.key_stages or (self.current_stage < len(self.key_stages)-1 and iteration > self.key_stages[self.current_stage+1]):
            if iteration in self.key_stages:
                reload = False
                current_stage = self.key_stages.index(iteration)
            else:
                reload = True
                current_stage = (np.array(self.key_stages) < iteration).sum() - 1
            # Reset scaling
            if not reload and self.scales_freedom != self.scales_freedom_list[current_stage]:
                scaling = self.get_scaling
                self._scaling.data = self.scaling_inverse_activation(scaling)
            # Update stage
            self.current_stage = current_stage
            # Reset acutance activation
            current_acutance = self.get_acutance
            self.acutance_min, self.acutance_max = self.acutance_interval_list[self.current_stage]
            self.acutance_activation = get_range_activation(self.acutance_min, self.acutance_max)
            self.inv_acutance_activation = get_range_inv_activation(self.acutance_min, self.acutance_max)
            if not reload and current_acutance.max() <= self.acutance_max and current_acutance.min() >= self.acutance_min:
                self._acutance.data = self.inv_acutance_activation(current_acutance)
                new_acutance = self.get_acutance
                print(f"Acutance update error: {torch.norm(current_acutance - new_acutance)} at iteration {iteration}!!!")
            # Reset other attributes
            self.scales_freedom         = self.scales_freedom_list[self.current_stage]
            self.densification_interval = self.densification_interval_list[self.current_stage]
            self.opacity_reset_interval = self.reset_opacity_interval_list[self.current_stage]

    @property
    def min_opacity_pruning(self):
        return self.final_min_opacity_pruning

    def regularization_loss(self, **kwargs):
        return super().regularization_loss(**kwargs)
    
    @property
    def get_acutance(self):
        return self.acutance_activation(self._acutance)
    
    @property
    def get_sharpen_opacity(self):
        if self.sharpen_alpha:
            k_acu = self.get_acutance
            opacity = self.get_opacity
            ax, bx = (1 + k_acu) / 4, (3 - k_acu) / 4
            y1 = (1 - k_acu) / (1 + k_acu) * opacity
            y2 = (1 + k_acu) / (1 - k_acu) * opacity - k_acu / (1 - k_acu)
            y3 = (1 - k_acu) / (1 + k_acu) * opacity + 2 * k_acu / (1 + k_acu)
            sharpen = torch.where(opacity < ax, y1, y2)
            sharpen = torch.where(opacity >= bx, y3, sharpen)
        else:
            sharpen = self.get_opacity
        return sharpen
    
    @property
    def get_normalized_acutance(self):
        acutance = self.get_acutance
        return (acutance - self.acutance_min) / (self.acutance_max - self.acutance_min)
    
    @property
    def get_scaling(self):
        if self.scales_freedom == 2:
            scaling = self.scaling_activation(self._scaling)
            s1, s2 = scaling[..., 0], scaling[..., 2]
            sh = (s1 * s2) / (s1**2 + s2**2)**.5 * 2**.5
            scaling = torch.stack([s1, sh, s2, sh, s1, sh, s2, sh], dim=-1)
        elif self.scales_freedom == 4:
            scaling = self.scaling_activation(self._scaling)
            s1, s2, s3, s4 = scaling[..., 0], scaling[..., 2], scaling[..., 4], scaling[..., 6]
            sh12 = (s1 * s2) / (s1**2 + s2**2)**.5 * 2**.5
            sh23 = (s2 * s3) / (s2**2 + s3**2)**.5 * 2**.5
            sh34 = (s3 * s4) / (s3**2 + s4**2)**.5 * 2**.5
            sh41 = (s4 * s1) / (s4**2 + s1**2)**.5 * 2**.5
            scaling = torch.stack([s1, sh12, s2, sh23, s3, sh34, s4, sh41], dim=-1)
        else:
            scaling = self.scaling_activation(self._scaling)
        return scaling
    
    @property
    def get_thetas(self):
        if self.scales_freedom == 2:
            thetas = (torch.sigmoid(torch.zeros_like(self._thetas)) + self.theta_residual).cumsum(dim=-1)
            thetas = thetas / thetas[..., -1:]
            return thetas
        else:
            thetas = (torch.sigmoid(self._thetas) + self.theta_residual).cumsum(dim=-1)
            thetas = thetas / thetas[..., -1:]
            return thetas

    @property
    def get_rotation(self):
        quat = super().get_rotation
        rotation = quaternion_to_matrix(quat)
        return rotation.reshape([-1, 9])

    @property
    def get_l1l2rates(self):
        if self.current_stage < self.l1l2rates_free_stage:
            return .5 * torch.ones_like(self._l1l2_rates)
        else:
            return torch.sigmoid(self._l1l2_rates)

    def get_normal(self, camera_center):
        rot_mat = self.get_rotation.reshape([-1, 3, 3])
        normal = rot_mat[..., :3, 2]
        dir = torch.nn.functional.normalize(camera_center - self.get_xyz, dim=-1)
        vis = (dir * normal).sum(dim=-1, keepdim=True) > 0
        normal = torch.where(vis, normal, -normal)
        return normal

    def create_from_pcd(self, *args, **kwargs):
        super().create_from_pcd(*args, **kwargs)
        acutances = self.inv_acutance_activation(.5 * torch.ones((self._xyz.shape[0], 1), dtype=torch.float, device="cuda")) if self.acutance_min >= 0. else self.inv_acutance_activation(torch.zeros((self._xyz.shape[0], 1), dtype=torch.float, device="cuda"))
        self._acutance = nn.Parameter(acutances.requires_grad_(True))
        self._rotation.data = torch.randn_like(self._rotation)
        self._scaling = nn.Parameter(self._scaling[..., :1].expand(self._scaling.shape[0], self.kernel_K).clone())
        self._scaling.data[:] = self._scaling[:, 0].median()
        self._thetas = nn.Parameter(torch.zeros_like(self._scaling))
        self._l1l2_rates = nn.Parameter(torch.zeros([self._thetas.shape[0], 1]).float().cuda())

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._acutance,
            self._thetas,
            self._l1l2_rates,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def training_setup(self, training_args):
        self.no_recenter = training_args.no_recenter
        self.no_resetopacity = training_args.no_resetopacity
        self.densification_interval = training_args.densification_interval
        self.opacity_reset_interval = training_args.opacity_reset_interval
        self.percent_dense = training_args.percent_drk_dense
        self.use_mcmc = getattr(training_args, "use_mcmc", False)
        self.mcmc_strategy = getattr(training_args, "mcmc_strategy", "replace")
        self.mcmc_start_iter = getattr(training_args, "mcmc_start_iter", -1)
        self.mcmc_end_iter = getattr(training_args, "mcmc_end_iter", -1)
        self.mcmc_cap_max = getattr(training_args, "mcmc_cap_max", -1)
        self.mcmc_growth_rate = getattr(training_args, "mcmc_growth_rate", 1.05)
        self.mcmc_grad_weight = getattr(training_args, "mcmc_grad_weight", 0.0)
        self.mcmc_scale_weight = getattr(training_args, "mcmc_scale_weight", 0.0)
        self.mcmc_min_opacity = getattr(training_args, "mcmc_min_opacity", 0.005)
        self.mcmc_noise_lr = getattr(training_args, "mcmc_noise_lr", 0.0)
        self.mcmc_opacity_reg = getattr(training_args, "mcmc_opacity_reg", 0.0)
        self.mcmc_scale_reg = getattr(training_args, "mcmc_scale_reg", 0.0)
        self.mcmc_prune_min_opacity = getattr(training_args, "mcmc_prune_min_opacity", 0.0)
        self.mcmc_prune_score = getattr(training_args, "mcmc_prune_score", "opacity")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.opacity_gradient_accum = torch.zeros_like(self.xyz_gradient_accum)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.cache_sort = training_args.cache_sort
        self.tile_culling = training_args.tile_culling
        if self.cache_sort:
            print("Cache Sort is enabled !!! This may slow down the training process but avoid popping artifacts. Not necessary for the final PSNR (with improvement around 0.1~0.2dB). Please turn it on if render with Mesh2DRK.")
        if self.tile_culling:
            print("Tile Culling is enabled!!! Recommended for inference with higher speed. Not recommended for training.")

        if training_args.is_unbounded:
            densify_grad_threshold = {'dense': 1e-3, 'middle': 1.5e-3, 'sparse': 2e-3}
            self.densify_grad_threshold = densify_grad_threshold[training_args.kernel_density]
            prune_threshold        = {'dense': 1e-1, 'middle': 1e-1,   'sparse': 1e-1}
        else:
            densify_grad_threshold = {'dense': 5e-4, 'middle': 1e-3, 'sparse': 2e-3}
            self.densify_grad_threshold = densify_grad_threshold[training_args.kernel_density]
            prune_threshold        = {'dense': 5e-2, 'middle': 5e-2, 'sparse': 1e-1}

        if training_args.specified_acu_range:
            acu_min, acu_max = training_args.specified_acu_min, training_args.specified_acu_max
            self.acutance_interval_list = [[acu_min, acu_max], [acu_min, acu_max], [acu_min, acu_max], [acu_min, acu_max]]
            print(f"Using specified acutance range: {acu_min}, {acu_max}")

        self.prune_threshold = prune_threshold[training_args.kernel_density]
        print('#'*50)
        print(f"Using {training_args.kernel_density} with densify_grad_threshold: {self.densify_grad_threshold}, prune_threshold: {self.prune_threshold}")
        print('#'*50)

        # Lower learning rate
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_drk_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_drk_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_drk_lr, "name": "rotation"},
            {'params': [self._acutance], 'lr': training_args.acutance_drk, "name": "acutance"},
            {'params': [self._thetas], 'lr': training_args.thetas_drk, "name": "thetas"},
            {'params': [self._l1l2_rates], 'lr': training_args.l1l2rates_drk, "name": "l1l2_rates"},
        ]
        self.acu_scheduler_args       = get_expon_lr_func(lr_init=training_args.acutance_drk, lr_final=training_args.acutance_drk_final, lr_delay_mult=training_args.position_lr_delay_mult, max_steps=training_args.position_lr_max_steps)
        self.l1l2rates_scheduler_args = get_expon_lr_func(lr_init=training_args.l1l2rates_drk, lr_final=training_args.l1l2rates_drk_final, lr_delay_mult=training_args.position_lr_delay_mult,  max_steps=training_args.position_lr_max_steps)
        self.scales_scheduler_args    = get_expon_lr_func(lr_init=training_args.scaling_drk_lr, lr_final=training_args.scaling_drk_lr_final, lr_delay_mult=training_args.position_lr_delay_mult, max_steps=training_args.position_lr_max_steps)
        self.thetas_scheduler_args    = get_expon_lr_func(lr_init=training_args.thetas_drk, lr_final=training_args.thetas_drk_final, lr_delay_mult=training_args.position_lr_delay_mult, max_steps=training_args.position_lr_max_steps)
        self.rotation_scheduler_args    = get_expon_lr_func(lr_init=training_args.rotation_drk_lr, lr_final=training_args.rotation_drk_lr_final, lr_delay_mult=training_args.position_lr_delay_mult, max_steps=training_args.position_lr_max_steps)

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args       = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale, lr_final=training_args.position_lr_final*self.spatial_lr_scale, lr_delay_mult=training_args.position_lr_delay_mult, max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                self.current_xyz_lr = lr
            if param_group["name"] == "acutance":
                lr = self.acu_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "scaling":
                lr = self.scales_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "thetas":
                lr = self.thetas_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "l1l2_rates":
                lr = self.l1l2rates_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group['lr'] = lr

    def add_densification_stats(self, viewspace_point_tensor, update_filter, opacity_tensor=None):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        if opacity_tensor is not None:
            self.opacity_gradient_accum[update_filter] += torch.norm(opacity_tensor.grad[update_filter], dim=-1, keepdim=True)
    
    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz           = optimizable_tensors["xyz"]
        self._features_dc   = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity       = optimizable_tensors["opacity"]
        self._acutance      = optimizable_tensors["acutance"]
        self._scaling       = optimizable_tensors["scaling"]
        self._rotation      = optimizable_tensors["rotation"]
        self._thetas        = optimizable_tensors["thetas"]
        self._l1l2_rates    = optimizable_tensors["l1l2_rates"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.opacity_gradient_accum = self.opacity_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_acutances, new_scaling, new_rotation, new_thetas, new_l1l2rates):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "acutance": new_acutances,
            "scaling" : new_scaling,
            "rotation" : new_rotation,
            "thetas": new_thetas,
            "l1l2_rates": new_l1l2rates
            }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz           = optimizable_tensors["xyz"]
        self._features_dc   = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity       = optimizable_tensors["opacity"]
        self._acutance      = optimizable_tensors["acutance"]
        self._scaling       = optimizable_tensors["scaling"]
        self._rotation      = optimizable_tensors["rotation"]
        self._thetas        = optimizable_tensors["thetas"]
        self._l1l2_rates    = optimizable_tensors["l1l2_rates"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.opacity_gradient_accum = torch.zeros_like(self.xyz_gradient_accum)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def reset_opacity(self):
        if self.no_resetopacity:
            return
        # Reset Opacity to close 0
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        # Reset Acutance
        acutance_new = self.inv_acutance_activation(torch.ones_like(self.get_opacity)*(self.acutance_max-self.acutance_min)*.5 + self.acutance_min)
        optimizable_tensors = self.replace_tensor_to_optimizer(acutance_new, "acutance")
        self._acutance = optimizable_tensors["acutance"]
        # Recenter
        if not self.no_recenter:
            self.recenter()

    def refresh_parameters_from_optimizer(self, optimizable_tensors):
        super().refresh_parameters_from_optimizer(optimizable_tensors)
        self._acutance = optimizable_tensors["acutance"]
        self._thetas = optimizable_tensors["thetas"]
        self._l1l2_rates = optimizable_tensors["l1l2_rates"]

    def _mcmc_updated_params(self, idxs, ratio):
        params = super()._mcmc_updated_params(idxs, ratio)
        params["acutance"] = self._acutance[idxs]
        params["thetas"] = self._thetas[idxs]
        params["l1l2_rates"] = self._l1l2_rates[idxs]
        return params

    def _mcmc_assign_params(self, indices, params):
        super()._mcmc_assign_params(indices, params)
        self._acutance.data[indices] = params["acutance"]
        self._thetas.data[indices] = params["thetas"]
        self._l1l2_rates.data[indices] = params["l1l2_rates"]

    def _mcmc_append_params(self, params):
        self.densification_postfix(
            params["xyz"],
            params["f_dc"],
            params["f_rest"],
            params["opacity"],
            params["acutance"],
            params["scaling"],
            params["rotation"],
            params["thetas"],
            params["l1l2_rates"],
        )

    @torch.no_grad()
    def add_mcmc_noise(self):
        if self.mcmc_noise_lr <= 0.0 or self.current_xyz_lr <= 0.0 or self.get_xyz.shape[0] == 0:
            return

        def op_sigmoid(x, k=100, x0=0.995):
            return 1 / (1 + torch.exp(-k * (x - x0)))

        rot = self.get_rotation.reshape(-1, 3, 3)
        tangent_scale = self.get_scaling.mean(dim=-1)
        local_noise = torch.randn_like(self._xyz)
        local_noise[:, :2] = local_noise[:, :2] * tangent_scale[:, None]
        local_noise[:, 2] = local_noise[:, 2] * (0.2 * tangent_scale)
        noise = torch.bmm(rot, local_noise.unsqueeze(-1)).squeeze(-1)
        noise = noise * op_sigmoid(1 - self.get_opacity) * self.mcmc_noise_lr * self.current_xyz_lr
        self._xyz.add_(noise)

    @torch.no_grad()
    def recenter(self):
        thetas = self.get_thetas * torch.pi * 2
        scales = self.get_scaling
        u = scales * torch.cos(thetas)  # [B, K]
        v = scales * torch.sin(thetas)  # [B, K]
        rot = self.get_rotation.reshape([-1, 3, 3])  # [B, 3, 3]
        vert = rot[:, None, :3, 0] * u[:, :, None] + rot[:, None, :3, 1] * v[:, :, None]  # [B, K, 3]
        center = vert.mean(dim=1)  # [B, 3]
        new_vert = vert - center[:, None]  # [B, K, 3]
        new_u = (new_vert * rot[:, None, :3, 0]).sum(dim=-1)  # [B, K]
        new_v = (new_vert * rot[:, None, :3, 1]).sum(dim=-1)  # [B, K]
        new_scales = torch.stack([new_u, new_v], dim=-1).norm(dim=-1).clip(1e-5)  # [B, K]
        new_thetas = torch.acos(new_u / new_scales) / (2 * torch.pi)  # [B, K]
        new_thetas[..., -1] = 1
        new_thetas_l = torch.cat([torch.zeros_like(new_thetas[..., :1]), new_thetas[..., :-1]], dim=-1)  # [B, K]
        delta_thetas = new_thetas - new_thetas_l  # [B, K]
        A = delta_thetas[..., None].expand([scales.shape[0], self.kernel_K, self.kernel_K]) - torch.eye(self.kernel_K, dtype=scales.dtype, device=scales.device)  # [B, K, K]
        b = self.theta_residual * (1 - self.kernel_K * delta_thetas)  # [B, K]
        new_per_theta_ = torch.einsum('nab,nb->na', torch.linalg.pinv(A), b)  # [B, K]
        new_per_theta = new_per_theta_.clip(.1, .9)  # [B, K]
        new_raw_theta = inverse_sigmoid(new_per_theta)  # [B, K]
        valid_mask = (delta_thetas > 0).all(dim=-1)[:, None].expand_as(new_raw_theta) & (new_per_theta_!=0).all(dim=-1)[:, None].expand_as(new_raw_theta)  # [B, K]
        new_raw_theta = torch.where(valid_mask, new_raw_theta, torch.zeros_like(new_raw_theta))
        self._xyz.data = self._xyz + center
        self._scaling.data = self.scaling_inverse_activation(new_scales)
        self._thetas.data = new_raw_theta

    def grad_postprocess(self):
        return

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        if self.clone_only:
            return
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1).mean(dim=-1, keepdim=True).repeat(1, 3)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        samples[..., -1] = 0.2 * samples[..., -1]
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling       = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation      = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc   = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity       = self._opacity[selected_pts_mask].repeat(N,1)
        new_acutances     = self._acutance[selected_pts_mask].repeat(N,1)
        new_thetas        = self._thetas[selected_pts_mask].repeat(N,1)
        new_l1l2rates     = self._l1l2_rates[selected_pts_mask].repeat(N,1)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_acutances, new_scaling, new_rotation, new_thetas, new_l1l2rates)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        if not self.clone_only:
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc   = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities     = inverse_sigmoid(1 - (1 - self.get_opacity[selected_pts_mask]))
        new_acutances     = self._acutance[selected_pts_mask]
        new_scaling       = self._scaling[selected_pts_mask]
        new_rotation      = self._rotation[selected_pts_mask]
        new_thetas        = self._thetas[selected_pts_mask]
        new_l1l2rates     = self._l1l2_rates[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_acutances, new_scaling, new_rotation, new_thetas, new_l1l2rates)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Opacity-gradient driven densification: combine position and opacity gradients
        opacity_grads = self.opacity_gradient_accum / self.denom
        opacity_grads[opacity_grads.isnan()] = 0.0
        combined_grads = grads + self.opacity_densify_grad_threshold * opacity_grads

        self.densify_and_clone(combined_grads, max_grad, extent)
        self.densify_and_split(combined_grads, max_grad, extent)

        prune_mask = (self.get_sharpen_opacity < min_opacity).squeeze()
        # Visibility-aware pruning: also prune Gaussians with low visibility
        if self.denom.shape[0] == prune_mask.shape[0]:
            visibility_ratio = self.denom.squeeze() / max(self.iteration - self.key_stages[0], 1)
            low_visibility = visibility_ratio < 0.02
            low_opacity = (self.get_sharpen_opacity < 0.1).squeeze()
            floater_mask = torch.logical_and(low_visibility, low_opacity)
            prune_mask = torch.logical_or(prune_mask, floater_mask)
        if max_screen_size:
            if extent == 0.0:
                extent = 1e2
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        l.append('acutance')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._thetas.shape[1]):
            l.append('theta_{}'.format(i))
        for i in range(self._l1l2_rates.shape[1]):
            l.append('l1l2rate_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        acutances = self._acutance.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        theta = self._thetas.detach().cpu().numpy()
        l1l2rate = self._l1l2_rates.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, acutances, scale, rotation, theta, l1l2rate), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        acutances = np.asarray(plydata.elements[0]["acutance"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        theta_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("theta")]
        theta_names = sorted(theta_names, key = lambda x: int(x.split('_')[-1]))
        thetas = np.zeros((xyz.shape[0], len(theta_names)))
        for idx, attr_name in enumerate(theta_names):
            thetas[:, idx] = np.asarray(plydata.elements[0][attr_name])

        l1l2rate_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("l1l2rate")]
        l1l2rate_names = sorted(l1l2rate_names, key = lambda x: int(x.split('_')[-1]))
        l1l2_rates = np.zeros((xyz.shape[0], len(l1l2rate_names)))
        for idx, attr_name in enumerate(l1l2rate_names):
            l1l2_rates[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._acutance = nn.Parameter(torch.tensor(acutances, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._thetas = nn.Parameter(torch.tensor(thetas, dtype=torch.float, device="cuda").requires_grad_(True))
        self._l1l2_rates = nn.Parameter(torch.tensor(l1l2_rates, dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
