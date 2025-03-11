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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float * depths,
	const float* scales,
	const float* thetas,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float* means3D,
	const uint32_t* tiles_touched,
	const float focal_x,
	const float focal_y,
	const float * rotations,
	const float2* points_xy_image,
	float W,
	float H,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	bool tile_culling,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;
		uint32_t left_tile_num = tiles_touched[idx];

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);
		
		// Project vertices of the kernel onto the image plane
		float vert2d[KERNEL_K][2] = {0.f};
		if(tile_culling) {
			float3 vert_center_world = {means3D[3 * idx + 0], means3D[3 * idx + 1], means3D[3 * idx + 2]};
			float3 vert_center_view = transformPoint4x3(vert_center_world, viewmatrix);
			scales = scales + idx * KERNEL_K;
			thetas = thetas + idx * KERNEL_K;
			const float * Rs = rotations + idx * 9;
			for(int ii=0; ii<KERNEL_K; ii++) {
				float scale = scales[ii];
				float theta = ii==0? 0: thetas[ii-1];
				float u = NEAREST_VERT_RADIUS * scale * cosf(theta * 2.f * PI);
				float v = NEAREST_VERT_RADIUS * scale * sinf(theta * 2.f * PI);
				float3 vert_world = {
					means3D[3 * idx + 0] + u * Rs[0] + v * Rs[1], 
					means3D[3 * idx + 1] + u * Rs[3] + v * Rs[4], 
					means3D[3 * idx + 2] + u * Rs[6] + v * Rs[7]};
				
				float3 vert_view = transformPoint4x3(vert_world, viewmatrix);
				adjustVertView(vert_center_view, vert_view);
				float safe_z = copysignf(0.0000001f + fabsf(vert_view.z), vert_view.z);
				float w = vert_view.x / safe_z * focal_x + (W - 1) / 2.0f;
				float h = vert_view.y / safe_z * focal_y + (H - 1) / 2.0f;
				vert2d[ii][0] = w;
				vert2d[ii][1] = h;
			}
		}		

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				// Check if the tile is visible
				float minX = (float) (x * BLOCK_X) - .5f - 0.000001f;
				float minY = (float) (y * BLOCK_Y) - .5f - 0.000001f;
				float maxX = (float) (x * BLOCK_X + BLOCK_X - 1) + .5f + 0.000001f;
				float maxY = (float) (y * BLOCK_Y + BLOCK_Y - 1) + .5f + 0.000001f;

				if(tile_culling) {
					bool visible = doesPolygonIntersectAABB(vert2d, minX, minY, maxX, maxY);
					if (!visible) {
						continue;
					}
				}

				// Avoid float error causes overflow
				if(left_tile_num == 0)
					break;
				left_tile_num--;

				float ray_t = 0.f;
				if(PRESORT_WITH_NEAREST_PIXEL || PRESORT_WITH_AVG_VALID_PIXEL){
					// Use nearest corner pixel to pre-sort
					float mean_ray_t = 0.f;
					float valid_count = 1.0f;
					ray_t = 10000000000.f;
					float2 pixf[4];
					pixf[0].x = minX; pixf[0].y = minY;
					pixf[1].x = maxX; pixf[1].y = minY;
					pixf[2].x = minX; pixf[2].y = maxY;
					pixf[3].x = maxX; pixf[3].y = maxY;
					for(int ii=0; ii<4; ii++) {
						float3 rayt_zscale_cosndir = calculate_rayt(viewmatrix, cam_pos, means3D+idx*3, focal_x, focal_y, rotations+idx*9, pixf[ii], W, H, true);
						float ray_t_corner = rayt_zscale_cosndir.x;
						if(ray_t_corner < ray_t)
							ray_t = max(0.01f, ray_t_corner);
						if(ray_t_corner > 0.f && ray_t_corner < 100.f) {
							mean_ray_t += ray_t;
							valid_count += 1.0f;
						}
					}
					if(PRESORT_WITH_AVG_VALID_PIXEL)
						ray_t = valid_count > 0.f? (mean_ray_t / valid_count): depths[idx];
				}
				else if (PRESORT_WITH_CENTER_PIXEL){
					// Use the center pixel to pre-sort
					float2 pixf = {(float) (x * BLOCK_X + BLOCK_X / 2), (float) (y * BLOCK_Y + BLOCK_Y / 2)};
					float3 rayt_zscale_cosndir = calculate_rayt(viewmatrix, cam_pos, means3D+idx*3, focal_x, focal_y, rotations+idx*9, pixf, W, H, true);
					ray_t = rayt_zscale_cosndir.x;
				}
				else if(PRESORT_WITH_CLOSEST_PIXEL) {
					const float2 point2d  = points_xy_image[idx];
					const float2 closest_pixel = {max(minX, min(maxX, point2d.x)), max(minY, min(maxY, point2d.y))};
					float3 rayt_zscale_cosndir = calculate_rayt(viewmatrix, cam_pos, means3D+idx*3, focal_x, focal_y, rotations+idx*9, closest_pixel, W, H, true);
					ray_t = rayt_zscale_cosndir.x;
				}
				else{ // PRESORT_WITH_CENTER_VERT
					ray_t = depths[idx];
				}

				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&ray_t);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}

		// Fill in missing tiles with a dummy key/value pair
		while(left_tile_num > 0) {
			uint64_t key = rect_min.y * grid.x + rect_min.x;
			key <<= 32;
			float ray_t = 100000000.f;
			key |= *((uint32_t*)&ray_t);
			gaussian_keys_unsorted[off] = key;
			gaussian_values_unsorted[off] = -1;
			off++;
			left_tile_num--;
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}


CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	
	obtain(chunk, geom.op, P * 3, 128);
	obtain(chunk, geom.op_tu, P, 128);
	obtain(chunk, geom.op_tv, P, 128);
	obtain(chunk, geom.op_n, P, 128);

	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
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
	int* radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess2D(
		P, D, M,
		means3D,
		scales,
		thetas,
		opacities,
		acutances,
		scale_modifier,
		rotations,
		shs,
		geomState.clamped,
		colors_precomp,
		viewmatrix, projmatrix,
		cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.rgb,
		tile_grid,
		geomState.tiles_touched,

		geomState.op,
		geomState.op_tu,
		geomState.op_tv,
		geomState.op_n,

		prefiltered,
		tile_culling
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		scales,
		thetas,
		viewmatrix,
		projmatrix,
		cam_pos,
		means3D,
		geomState.tiles_touched,
		focal_x,
		focal_y, rotations,
		geomState.means2D,
		(float) width, (float) height,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_culling,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		means3D,
		geomState.means2D,
		feature_ptr,
		geomState.depths,
		
		rotations,
		scales,
		thetas,
		l1l2_rates,
		opacities,
		acutances,
		cam_pos,
		focal_x, focal_y,
		viewmatrix,

		geomState.op,
		geomState.op_tu,
		geomState.op_tv,
		geomState.op_n,

		out_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_depth,
		out_normal,
		cache_sort), debug)

	return num_rendered;
}


// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
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
	char* img_buffer,

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
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;

	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		color_ptr,
		geomState.depths,

		means3D,
		rotations,
		scales,
		thetas,
		l1l2_rates,
		opacities,
		acutances,
		cam_pos,
		focal_x, focal_y,
		viewmatrix,

		geomState.op,
		geomState.op_tu,
		geomState.op_tv,
		geomState.op_n,

		out_alpha,
		out_color,
		out_depth,
		out_normal,

		imgState.n_contrib,
		dL_dpix,
		dL_dpix_alpha,
		dL_ddepths,
		dL_dnormal,
		(float3*)dL_dmean2D,
		(float3*)dL_dmean2D_densify,
		(float*)dL_dopacity_densify,
		dL_dopacity,
		dL_dcolor,
		dL_dacutance,

		dL_dmean3D,
		dL_drot,
		dL_dscale,
		dL_dthetas,
		dL_dl1l2_rates,
		
		cache_sort), debug)

	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)cam_pos,
		dL_dcolor,
		dL_dsh,
		dL_dmean3D), debug)
}
