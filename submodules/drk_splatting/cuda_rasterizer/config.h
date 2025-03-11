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

#define KERNEL_K 8 // Default 8 or 16

#define PI 3.14159265359

#define SHARPEN_ALPHA false

#define CACHE_SIZE 16

#define LOW_PASS_FILTER true

#define FARTHEST_DISTANCE 100.f

#define NEAREST_VERT_RADIUS 2.5
#define PRESORT_WITH_NEAREST_PIXEL   false    // Pre-sort with the nearest intersection between tile rays and the kernel
#define PRESORT_WITH_AVG_VALID_PIXEL false    // Pre-sort with the average distance between tile rays and the kernel
#define PRESORT_WITH_CENTER_PIXEL    false    // Pre-sort with the depth of the center pixel
#define PRESORT_WITH_CLOSEST_PIXEL   false    // Pre-sort with the depth of the pixel closet to the 2D DRK center
// #define PRESORT_WITH_CENTER_VERT     false     // Pre-sort with the kernel center, the same as Gaussian Splatting

#endif