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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    means2D_densify,
    opacity_densify,
    sh,
    colors_precomp,
    opacities,
    scales,
    thetas,
    l1l2_rates,
    rotations,
    acutances,
    cache_sort,
    tile_culling,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        means2D_densify,
        opacity_densify,
        sh,
        colors_precomp,
        opacities,
        scales,
        thetas,
        l1l2_rates,
        rotations,
        acutances,
        cache_sort,
        tile_culling,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        means2D_densify,
        opacity_densify,
        sh,
        colors_precomp,
        opacities,
        scales,
        thetas,
        l1l2_rates,
        rotations,
        acutances,
        cache_sort,
        tile_culling,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            thetas,
            l1l2_rates,
            rotations,
            acutances,
            raster_settings.scale_modifier,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            cache_sort,
            tile_culling,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, normal, alpha, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, normal, alpha, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, opacities, means3D, scales, rotations, acutances, thetas, l1l2_rates, radii, sh, geomBuffer, binningBuffer, imgBuffer, color, alpha, depth, normal)
        ctx.cache_sort = cache_sort
        ctx.tile_culling = tile_culling
        # return color, radii
        return color, radii, depth, normal, alpha

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_radii, grad_out_depth, grad_out_normal, grad_out_alpha):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, opacities, means3D, scales, rotations, acutances, thetas, l1l2_rates, radii, sh, geomBuffer, binningBuffer, imgBuffer, color, alpha, depth, normal = ctx.saved_tensors
        cache_sort = ctx.cache_sort
        tile_culling = ctx.tile_culling

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp,
                opacities,
                scales,
                thetas,
                l1l2_rates,
                rotations, 
                acutances, 
                raster_settings.scale_modifier, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 

                color,
                alpha,
                depth,
                normal,

                grad_out_color,
                grad_out_alpha,
                grad_out_depth,
                grad_out_normal,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,

                cache_sort,
                tile_culling,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_means2D_densify, grad_opacity_densify, grad_colors_precomp, grad_opacities, grad_means3D, grad_sh, grad_scales, grad_thetas, grad_l1l2_rates, grad_rotations, grad_acutances = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_means2D_densify, grad_opacity_densify, grad_colors_precomp, grad_opacities, grad_means3D, grad_sh, grad_scales, grad_thetas, grad_l1l2_rates, grad_rotations, grad_acutances = _C.rasterize_gaussians_backward(*args)
        
        grads = (
            grad_means3D,
            grad_means2D,
            grad_means2D_densify,
            grad_opacity_densify,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_thetas,
            grad_l1l2_rates,
            grad_rotations,
            grad_acutances,
            None,
            None,
            None
        )

        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, means2D_densify, opacity_densify, opacities, shs = None, colors_precomp = None, scales=None, thetas=None, l1l2_rates=None, rotations=None, acutances=None, cache_sort=False, tile_culling=False):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            means2D_densify,
            opacity_densify,
            shs,
            colors_precomp,
            opacities,
            scales,
            thetas,
            l1l2_rates,
            rotations,
            acutances,
            cache_sort,
            tile_culling,
            raster_settings, 
        )