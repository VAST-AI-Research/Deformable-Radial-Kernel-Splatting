/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"
#include "cuda_rasterizer/config.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("get_compiled_kernel_k", []() { return KERNEL_K; });
  m.def("get_compiled_low_pass_filter", []() { return bool(LOW_PASS_FILTER); });
  m.def("get_compiled_force_l1_kernel", []() { return bool(DRK_FORCE_L1_KERNEL); });
  m.def("get_compiled_force_acutance_one_kernel", []() { return bool(DRK_FORCE_ACUTANCE_ONE_KERNEL); });
  m.def("get_compiled_screen_space_l1_kernel", []() { return bool(DRK_SCREEN_SPACE_L1_KERNEL); });
  m.def("get_compiled_tight_exact_aabb", []() { return bool(DRK_TIGHT_EXACT_AABB); });
  m.def("get_compiled_tight_exact_aabb_margin", []() { return float(DRK_TIGHT_EXACT_AABB_MARGIN); });
  m.def("get_compiled_pixel_center_offset_x", []() { return float(DRK_PIXEL_CENTER_OFFSET_X); });
  m.def("get_compiled_pixel_center_offset_y", []() { return float(DRK_PIXEL_CENTER_OFFSET_Y); });
  m.def("get_compiled_pixel_center_offset_apply_projection", []() { return bool(DRK_PIXEL_CENTER_OFFSET_APPLY_PROJECTION); });
  m.def("get_compiled_pixel_center_offset_apply_ray", []() { return bool(DRK_PIXEL_CENTER_OFFSET_APPLY_RAY); });
  // m.def("rasterize_aussians_filter", &RasterizeGaussiansfilterCUDA);
  // m.def("mark_visible", &markVisible);

}
