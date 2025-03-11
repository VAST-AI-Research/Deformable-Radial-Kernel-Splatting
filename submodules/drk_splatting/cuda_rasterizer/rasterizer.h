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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float* thetas,
			const float* l1l2_rates,
			const float scale_modifier,
			const float* rotations,
			const float* acutances,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* out_depth,
			float* out_normal,
			float* out_alpha,
			bool cache_sort,
			bool tile_culling,
			int* radii = nullptr,
			bool debug = false);
		
		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,

			const float* scales,
			const float* thetas,
			const float* l1l2_rates,

			const float scale_modifier,
			const float* rotations,
			const float* acutances,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,

			const float* out_color,
			const float* out_alpha,
			const float* out_depth,
			const float* out_normal,

			const float* dL_dpix,
			const float* dL_dpix_alpha,
			const float* dL_ddepths,
			const float* dL_dnormal,
			float* dL_dmean2D,
			float* dL_dmean2D_densify,
			float* dL_dopacity_densify,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dsh,

			float* dL_dscale,
			float* dL_dthetas,
			float* dL_dl1l2_rates,

			float* dL_drot,
			float* dL_dacutance,

			bool cache_sort,
			bool debug);

	};
};

#endif