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
from drk_splatting import (
    GaussianRasterizationSettings as DRKRasterizationSettings,
    GaussianRasterizer as DRKRasterizer,
    make_rasterize_gaussians_forward_args as drk_make_rasterize_forward_args,
    rasterize_gaussians_forward as drk_rasterize_forward,
    rasterize_gaussians_forward_from_args as drk_rasterize_forward_from_args,
)
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


def _resolve_sh_degree(pc, sh_degree=None):
    degree = pc.active_sh_degree if sh_degree is None else sh_degree
    return max(0, min(int(degree), int(pc.active_sh_degree), int(pc.max_sh_degree), 4))


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
    sh_degree = _resolve_sh_degree(pc, sh_degree)

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
            # The CUDA rasterizers implement SH bases through degree 4.
            # Keep Python precomputed colors metric-equivalent.
            sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)
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


def drk_render_func(viewpoint_camera, pc, pipe=default_pipe, bg_color : torch.Tensor=None, scaling_modifier=1.0, override_color=None, vis_scale_rate=1., vis_acutance_rate=None, vis_l1l2rate_rate=None, vis_opacity_rate=None, opaque_mode=False, collect_densify=True, render_aux=True, return_radii=True, **kwargs):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    need_grad = torch.is_grad_enabled()
    collect_densify = kwargs.get("collect_densify", collect_densify)
    render_aux = kwargs.get("render_aux", render_aux)
    return_radii = kwargs.get("return_radii", return_radii)
    return_forward_args = kwargs.get("return_forward_args", False)
    cache_sort = kwargs.get("cache_sort", pc.cache_sort)
    tile_culling = kwargs.get("tile_culling", pc.tile_culling)
    sh_degree = _resolve_sh_degree(pc, kwargs.get("sh_degree", None))
    inference_cache = None
    if need_grad and return_forward_args:
        raise ValueError("return_forward_args is only valid for no-grad DRK inference")
    if (
        not need_grad
        and not opaque_mode
        and override_color is None
        and not pipe.convert_SHs_python
        and vis_scale_rate == 1.0
        and hasattr(pc, "get_inference_cache")
    ):
        inference_cache = pc.get_inference_cache()
    if need_grad and collect_densify:
        tensor_ref = inference_cache["xyz"] if inference_cache is not None else pc.get_xyz
        # Create zero tensors used by the densification gradients during training.
        screenspace_points = torch.nn.Parameter(torch.zeros_like(tensor_ref, dtype=tensor_ref.dtype, requires_grad=True, device="cuda"))
        means2D_densify = torch.nn.Parameter(torch.zeros_like(tensor_ref, dtype=tensor_ref.dtype, requires_grad=True, device="cuda"))
        opacity_grad_densify = torch.nn.Parameter(torch.zeros_like(tensor_ref[..., :1], dtype=tensor_ref.dtype, requires_grad=True, device="cuda"))
        means2D = screenspace_points
    else:
        tensor_ref = inference_cache["xyz"] if inference_cache is not None else pc.get_xyz
        if inference_cache is not None and "empty_xyz" in inference_cache:
            means2D = inference_cache["empty_xyz"]
            means2D_densify = inference_cache["empty_xyz"]
            opacity_grad_densify = inference_cache["empty_opacity"]
        else:
            means2D = torch.empty((0, 3), dtype=tensor_ref.dtype, device=tensor_ref.device)
            means2D_densify = torch.empty((0, 3), dtype=tensor_ref.dtype, device=tensor_ref.device)
            opacity_grad_densify = torch.empty((0, 1), dtype=tensor_ref.dtype, device=tensor_ref.device)

    if bg_color is None:
        bg_color = (
            inference_cache["black_bg"]
            if inference_cache is not None and "black_bg" in inference_cache
            else torch.zeros([3], dtype=torch.float32, device=tensor_ref.device)
        )

    raster_settings = None
    raster_key = None
    if inference_cache is not None and "raster_settings" in inference_cache:
        raster_cache = inference_cache["raster_settings"]
        raster_key = (
            id(viewpoint_camera),
            int(viewpoint_camera.image_height),
            int(viewpoint_camera.image_width),
            float(viewpoint_camera.FoVx),
            float(viewpoint_camera.FoVy),
            float(scaling_modifier),
            int(sh_degree),
            id(bg_color),
            id(viewpoint_camera.world_view_transform),
            id(viewpoint_camera.full_proj_transform),
            id(viewpoint_camera.camera_center),
        )
        raster_settings = raster_cache.get(raster_key)
    if raster_settings is None:
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        drk_settings = DRKRasterizationSettings
        raster_settings = drk_settings(
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
            debug=False,
        )
        if inference_cache is not None and "raster_settings" in inference_cache:
            raster_cache[raster_key] = raster_settings

    means3D = inference_cache["xyz"] if inference_cache is not None else pc.get_xyz
    if inference_cache is not None:
        opacity = inference_cache["opacity"]
        if vis_opacity_rate is not None or opaque_mode:
            opacity_cache = inference_cache.setdefault("constant_opacity", {})
            opacity_key = 1.0 if opaque_mode and vis_opacity_rate is None else float(vis_opacity_rate)
            opacity = opacity_cache.get(opacity_key)
            if opacity is None:
                opacity = torch.empty_like(inference_cache["opacity"]).fill_(opacity_key)
                opacity_cache[opacity_key] = opacity
    else:
        if vis_opacity_rate is not None:
            opacity = torch.ones_like(pc.get_opacity) * vis_opacity_rate
        else:
            opacity = pc.get_opacity if not opaque_mode else torch.ones_like(pc.get_opacity)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    if inference_cache is not None:
        scales = inference_cache["scaling"]
        rotations = inference_cache["rotation"]
    else:
        scales = pc.get_scaling * vis_scale_rate
        rotations = pc.get_rotation

    # Joint pose refinement (learned bundle adjustment): rigidly transform primitive
    # centers + kernel frames by this camera's learnable SE3 delta. Equivalent to moving
    # the camera; gradients flow to the pose delta via d(loss)/d(means3D, rotations).
    if inference_cache is None and getattr(pc, 'pose_refine', None) is not None \
            and getattr(viewpoint_camera, 'pose_idx', -1) >= 0 and rotations is not None:
        means3D, rotations = pc.pose_refine.transform(viewpoint_camera.pose_idx, means3D, rotations)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if inference_cache is not None:
            view_color_cache = inference_cache.get("view_colors")
            if view_color_cache is not None:
                view_color_key = (
                    id(viewpoint_camera),
                    int(sh_degree),
                    id(viewpoint_camera.camera_center),
                )
                colors_precomp = view_color_cache.get(view_color_key)
                if colors_precomp is None:
                    shs_view = inference_cache["features"].transpose(1, 2).view(
                        -1,
                        3,
                        (pc.max_sh_degree + 1) ** 2,
                    )
                    dir_pp = means3D - viewpoint_camera.camera_center
                    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                    # The CUDA rasterizers implement SH bases through degree 4.
                    # Keep Python precomputed colors metric-equivalent.
                    sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)
                    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0).contiguous()
                    view_color_cache[view_color_key] = colors_precomp
            if colors_precomp is None:
                shs = inference_cache["features"]
        elif pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # The CUDA rasterizers implement SH bases through degree 4.
            # Keep Python precomputed colors metric-equivalent.
            sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if inference_cache is not None:
        acutance = inference_cache["acutance"]
        if vis_acutance_rate is not None:
            acutance_cache = inference_cache.setdefault("constant_acutance", {})
            acutance_key = float(vis_acutance_rate)
            acutance = acutance_cache.get(acutance_key)
            if acutance is None:
                acutance = torch.empty_like(inference_cache["acutance"]).fill_(acutance_key)
                acutance_cache[acutance_key] = acutance

        l1l2_rates = inference_cache["l1l2_rates"]
        if vis_l1l2rate_rate is not None:
            l1l2_cache = inference_cache.setdefault("constant_l1l2_rates", {})
            l1l2_key = float(vis_l1l2rate_rate)
            l1l2_rates = l1l2_cache.get(l1l2_key)
            if l1l2_rates is None:
                l1l2_rates = torch.empty_like(inference_cache["l1l2_rates"]).fill_(l1l2_key)
                l1l2_cache[l1l2_key] = l1l2_rates
        thetas = inference_cache["thetas"]
    else:
        acutance = pc.get_acutance if vis_acutance_rate is None else torch.ones_like(pc.get_acutance) * vis_acutance_rate
        l1l2_rates = pc.get_l1l2rates if vis_l1l2rate_rate is None else torch.ones_like(pc.get_l1l2rates) * vis_l1l2rate_rate
        thetas = pc.get_thetas

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if not need_grad:
        empty = (
            inference_cache["empty"]
            if inference_cache is not None and "empty" in inference_cache
            else torch.empty(0, dtype=tensor_ref.dtype, device=tensor_ref.device)
        )
        shs_arg = empty if shs is None else shs
        colors_arg = empty if colors_precomp is None else colors_precomp
        raster_forward_args = None
        if inference_cache is not None and "raster_forward_args" in inference_cache and raster_key is not None:
            raster_forward_cache = inference_cache["raster_forward_args"]
            raster_forward_key = raster_key + (
                bool(cache_sort),
                bool(tile_culling),
                bool(render_aux),
                bool(return_radii),
                None if vis_acutance_rate is None else float(vis_acutance_rate),
                None if vis_l1l2rate_rate is None else float(vis_l1l2rate_rate),
                None if vis_opacity_rate is None else float(vis_opacity_rate),
                bool(opaque_mode),
            )
            raster_forward_args = raster_forward_cache.get(raster_forward_key)
            if raster_forward_args is None:
                raster_forward_args = drk_make_rasterize_forward_args(
                    means3D,
                    shs_arg,
                    colors_arg,
                    opacity,
                    scales,
                    thetas,
                    l1l2_rates,
                    rotations,
                    acutance,
                    cache_sort,
                    tile_culling,
                    render_aux,
                    return_radii,
                    raster_settings,
                )
                raster_forward_cache[raster_forward_key] = raster_forward_args
        if return_forward_args:
            if raster_forward_args is None:
                raster_forward_args = drk_make_rasterize_forward_args(
                    means3D,
                    shs_arg,
                    colors_arg,
                    opacity,
                    scales,
                    thetas,
                    l1l2_rates,
                    rotations,
                    acutance,
                    cache_sort,
                    tile_culling,
                    render_aux,
                    return_radii,
                    raster_settings,
                )
            return {"forward_args": raster_forward_args}
        if raster_forward_args is not None:
            results = drk_rasterize_forward_from_args(raster_forward_args)
        else:
            results = drk_rasterize_forward(
                means3D,
                shs_arg,
                colors_arg,
                opacity,
                scales,
                thetas,
                l1l2_rates,
                rotations,
                acutance,
                cache_sort,
                tile_culling,
                render_aux,
                return_radii,
                raster_settings,
            )
    else:
        args = {
            "means3D": means3D,
            "means2D": means2D,
            "means2D_densify": means2D_densify,
            "opacity_densify": opacity_grad_densify,
            "shs": shs,
            "colors_precomp": colors_precomp,
            "opacities": opacity,
            "scales": scales,
            "thetas": thetas,
            "l1l2_rates": l1l2_rates,
            "rotations": rotations,
            "acutances": acutance,
            "cache_sort": cache_sort,
            "tile_culling": tile_culling,
            "collect_densify": collect_densify,
            "render_aux": render_aux,
        }
        rasterizer = DRKRasterizer(raster_settings=raster_settings)
        import os as _os
        if _os.environ.get("DRK_DUMP_ARGS"):
            import torch as _torch
            _torch.save({"args": args, "raster_settings": raster_settings},
                        _os.environ["DRK_DUMP_ARGS"])
            print("[DRK_DUMP_ARGS] saved rasterizer args to", _os.environ["DRK_DUMP_ARGS"], flush=True)
            raise SystemExit(0)
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
