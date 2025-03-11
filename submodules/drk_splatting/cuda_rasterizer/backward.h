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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* bg_color,
		const float2* means2D,
		const float* colors,
		const float* depths,

		const float* means3D,
		const float* rotations,
		const float* scales,
		const float* thetas,
		const float* l1l2_rates,
		const float* opacities,
		const float* acutances,
		const float* cam_pos,
		const float focal_x, float focal_y,
		const float* viewmatrix,

		const float * op,
		const float * op_tu,
		const float * op_tv,
		const float * op_n,

		const float* final_Ts,
		const float* final_Colors,
		const float* final_Depths,
		const float* final_Normals,

		const uint32_t* n_contrib,
		const float* dL_dpixels,
		const float* dL_dpixels_alpha,
		const float* dL_ddepths,
		const float* dL_dnormal,
		float3* dL_dmean2D,
		float3* dL_dmean2D_densify,
		float* dL_dopacity_densify,
		float* dL_dopacity,
		float* dL_dcolors,
		float* dL_dacutance,

		float* dL_dmeans3D,
		float* dL_drotations,
		float* dL_dscale,
		float* dL_dthetas,
		float* dL_dl1l2_rates,
		bool cache_sort);

	void preprocess(
		int P, int D, int M,
		const float3* means,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec3* campos,
		float* dL_dcolor,
		float* dL_dsh,
		float* dL_dmeans3D);
}

#endif