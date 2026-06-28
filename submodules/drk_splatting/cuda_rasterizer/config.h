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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB

// // Debug // For accurate depth sorting per pixel
// #define BLOCK_X 1
// #define BLOCK_Y 1

#define BLOCK_X 16
#define BLOCK_Y 16

#ifndef KERNEL_K
#define KERNEL_K 8 // Default 8 or 16
#endif

#define PI 3.14159265359

#define SHARPEN_ALPHA false

#define CACHE_SIZE 16

// Bounded order-independent transparency: cap the number of CONTRIBUTING layers
// composited per pixel (front-to-back; with cache_sort this is the nearest-N).
// 0 = unlimited (original behaviour). >0 = hard cap (e.g. 8 for mobile mesh OIT).
#ifndef DRK_MAX_LAYERS
#define DRK_MAX_LAYERS 0
#endif

#ifndef LOW_PASS_FILTER
#define LOW_PASS_FILTER 1
#endif

#ifndef DRK_FORCE_L1_KERNEL
#define DRK_FORCE_L1_KERNEL 0
#endif

#ifndef DRK_FORCE_ACUTANCE_ONE_KERNEL
#define DRK_FORCE_ACUTANCE_ONE_KERNEL 0
#endif

#ifndef DRK_SCREEN_SPACE_L1_KERNEL
#define DRK_SCREEN_SPACE_L1_KERNEL 0
#endif

#ifndef DRK_TIGHT_EXACT_AABB
#define DRK_TIGHT_EXACT_AABB 0
#endif

#ifndef DRK_TIGHT_EXACT_AABB_MARGIN
#define DRK_TIGHT_EXACT_AABB_MARGIN 2.0f
#endif

// Tight projected-polygon AABB for the general (soft / low-pass) kernel path.
// Faster than the circumscribing-circle getRect, but (like tile_culling) it is
// not bit-identical under cache_sort, so it is an opt-in inference accelerator,
// OFF by default for bit-exact training.
#ifndef DRK_TIGHT_GENERAL_AABB
#define DRK_TIGHT_GENERAL_AABB 0
#endif

#ifndef DRK_PIXEL_CENTER_OFFSET_X
#define DRK_PIXEL_CENTER_OFFSET_X 0.0f
#endif

#ifndef DRK_PIXEL_CENTER_OFFSET_Y
#define DRK_PIXEL_CENTER_OFFSET_Y 0.0f
#endif

#ifndef DRK_PIXEL_CENTER_OFFSET_APPLY_PROJECTION
#define DRK_PIXEL_CENTER_OFFSET_APPLY_PROJECTION 1
#endif

#ifndef DRK_PIXEL_CENTER_OFFSET_APPLY_RAY
#define DRK_PIXEL_CENTER_OFFSET_APPLY_RAY 1
#endif

#define FARTHEST_DISTANCE 100.f

#define NEAREST_VERT_RADIUS 2.5
#define PRESORT_WITH_NEAREST_PIXEL   false    // Pre-sort with the nearest intersection between tile rays and the kernel
#define PRESORT_WITH_AVG_VALID_PIXEL false    // Pre-sort with the average distance between tile rays and the kernel
#define PRESORT_WITH_CENTER_PIXEL    false    // Pre-sort with the depth of the center pixel
#define PRESORT_WITH_CLOSEST_PIXEL   false    // Pre-sort with the depth of the pixel closet to the 2D DRK center
// #define PRESORT_WITH_CENTER_VERT     false     // Pre-sort with the kernel center, the same as Gaussian Splatting

#endif
