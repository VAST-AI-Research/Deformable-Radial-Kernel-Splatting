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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess2D(int P, int D, int M,
		const float* means3D,
		const float* scales,
		const float* thetas,
		const float* opacities,
		const float* acutances,
		const float scale_modifier,
		const float* rotations,
		const float* shs,
		bool* clamped,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const float* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* means2D,
		float* depths,
		float* rgb,
		const dim3 grid,
		uint32_t* tiles_touched,

		float * op,
		float * op_tu,
		float * op_tv,
		float * op_n,
		
		bool prefiltered,
		bool tile_culling);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* means3D,
		const float2* means2D,
		const float* colors,
		const float* depths,

		const float* rotations,
		const float* scales,
		const float* thetas,
		const float* l1l2_rates,
		const float* opacities,
		const float* acutances,
		const float* cam_pos,
		const float focal_x, float focal_y,
		const float* viewmatrix,

		float * op,
		float * op_tu,
		float * op_tv,
		float * op_n,
		
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* out_depth,
		float* out_normal,
		bool cache_sort);
}


#endif