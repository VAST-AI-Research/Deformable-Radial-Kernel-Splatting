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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from drk_splatting import GaussianRasterizationSettings as DRKRasterizationSettings, GaussianRasterizer as DRKRasterizer
from utils.sh_utils import eval_sh


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


class DefaultPipe:
    def __init__(self) -> None:
        self.convert_SHs_python = False
        self.debug = False
        self.compute_cov3D_python = False
default_pipe = DefaultPipe()


def render(viewpoint_camera, pc, pipe=default_pipe, bg_color : torch.Tensor=None, scaling_modifier = 1.0, override_color = None, vis_scale_rate=1., sh_degree=None, **kwargs):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_densify = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_densify.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    bg_color = torch.zeros([3], dtype=torch.float32, device=pc.get_xyz.device) if bg_color is None else bg_color
    sh_degree = pc.active_sh_degree if sh_degree is  None else sh_degree

    grs = GaussianRasterizationSettings
    gr = GaussianRasterizer
    raster_settings = grs(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    rasterizer = gr(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    means2D_densify = screenspace_points_densify
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling * vis_scale_rate
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        means2D_densify = means2D_densify,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points_densify": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth}


def drk_render_func(viewpoint_camera, pc, pipe=default_pipe, bg_color : torch.Tensor=None, scaling_modifier=1.0, override_color=None, vis_scale_rate=1., vis_acutance_rate=None, vis_l1l2rate_rate=None, opaque_mode=False, **kwargs):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.nn.Parameter(torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"))
    means2D_densify = torch.nn.Parameter(torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"))
    opacity_grad_densify = torch.nn.Parameter(torch.zeros_like(pc.get_xyz[..., :1], dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"))

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    bg_color = torch.zeros([3], dtype=torch.float32, device=pc.get_xyz.device) if bg_color is None else bg_color

    camera_center = viewpoint_camera.world_view_transform.T.inverse()[:3, 3]

    drk_settings = DRKRasterizationSettings
    rasterizer = DRKRasterizer

    raster_settings = drk_settings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = rasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity if not opaque_mode else torch.ones_like(pc.get_opacity)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = pc.get_scaling * vis_scale_rate
    rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    acutance = pc.get_acutance if vis_acutance_rate is None else torch.ones_like(pc.get_acutance) * vis_acutance_rate
    l1l2_rates = pc.get_l1l2rates if vis_l1l2rate_rate is None else torch.ones_like(pc.get_l1l2rates) * vis_l1l2rate_rate

    args = {"means3D": means3D, "means2D": means2D, "means2D_densify": means2D_densify, "opacity_densify": opacity_grad_densify, "shs": shs, "colors_precomp": colors_precomp, "opacities": opacity, "scales": scales, "thetas": pc.get_thetas, "l1l2_rates": l1l2_rates, "rotations": rotations, "acutances": acutance, 'cache_sort': pc.cache_sort, 'tile_culling': pc.tile_culling}
    if 'cache_sort' in kwargs:
        args['cache_sort'] = kwargs['cache_sort']
    if 'tile_culling' in kwargs:
        args['tile_culling'] = kwargs['tile_culling']
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    results = rasterizer(**args)
    rendered_image, radii, depth, normal, alpha = results

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "alpha": alpha,
            "viewspace_points_densify": means2D_densify,
            "opacity_grad_densify": opacity_grad_densify,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth,
            "normal": normal,
            "bg_color": bg_color}

