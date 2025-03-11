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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}


// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ colors,
	const float* __restrict__ depths,

	const float* __restrict__ means3D,
	const float* __restrict__ rotations,
	const float* __restrict__ scales,
	const float* __restrict__ thetas,
	const float* __restrict__ l1l2_rates,
	const float* __restrict__ opacities,
	const float* __restrict__ acutances,
	const float* __restrict__ cam_pos,
	const float focal_x, float focal_y,
	const float* viewmatrix,

	const float * __restrict__ op,
	const float * __restrict__ op_tu,
	const float * __restrict__ op_tv,
	const float * __restrict__ op_n,

	const float* __restrict__ final_Ts,
	const float* __restrict__ final_Colors,
	const float* __restrict__ final_Depths,
	const float* __restrict__ final_Normals,

	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dpixels_alpha,

	const float* __restrict__ dL_ddepths,
	const float* __restrict__ dL_dnormal,
	float3* __restrict__ dL_dmean2D,
	float3* __restrict__ dL_dmean2D_densify,
	float* __restrict__ dL_dopacity_densify,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_dacutance,

	float* __restrict__ dL_dmeans3D,
	float* __restrict__ dL_drotations,
	float* __restrict__ dL_dscale,
	float* __restrict__ dL_dthetas,
	float* __restrict__ dL_dl1l2_rates,

	bool cache_sort)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];

	__shared__ float  collected_mean3D  [BLOCK_SIZE*3];
	__shared__ float  collected_Rs      [BLOCK_SIZE*9];
	__shared__ float  collected_opacity [BLOCK_SIZE];
	__shared__ float  collected_acutance[BLOCK_SIZE];
	__shared__ float  collected_l1l2rate[BLOCK_SIZE];
	__shared__ float  collected_op_tu   [BLOCK_SIZE];
	__shared__ float  collected_op_tv   [BLOCK_SIZE];
	__shared__ float  collected_op_n    [BLOCK_SIZE];
	__shared__ float  collected_op      [BLOCK_SIZE*3];
	__shared__ float  collected_scales  [BLOCK_SIZE*KERNEL_K];
	__shared__ float  collected_thetas  [BLOCK_SIZE*KERNEL_K];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors.
	const float T_final     = inside ? final_Ts[pix_id] : 0;
	float color_final[3];
	float normal_final[3];
	if(inside)
		for(int ii=0; ii<3; ii++) {
			color_final[ii]  = final_Colors[ii*H*W + pix_id] - T_final * bg_color[ii];
			normal_final[ii] = final_Normals[ii*H*W + pix_id];
		}
	const float depth_final = inside ? final_Depths[pix_id]: 0;

	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
	float dL_dpixel_alpha = inside? dL_dpixels_alpha[pix_id]: 0.;
	float dL_ddepth = inside? dL_ddepths[pix_id]: 0.;
	float dL_dpixel_normal[3] = {0.};
	if (inside)
		for (int i = 0; i < 3; i++)
			dL_dpixel_normal[i] = dL_dnormal[i * H * W + pix_id];

	float last_alpha = 0;

	float T = 1.0f;
	float Ci[C] = { 0 };
	float Ni[3] = {0};
	float Di = 0.;

	// CACHE Array to sort the surfels
	float ray_t_array[CACHE_SIZE] = {0};
	int js[CACHE_SIZE] = {0};

	uint32_t contributor = 0;
	const uint32_t last_contributor = inside? n_contrib[pix_id]: 0;

	// Gradient of pixel coordinate w.r.t. normalized
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Pre-compute ray direction
	glm::vec3 dir_camera = {(pixf.x - (W - 1.0) * 0.5) / focal_x, (pixf.y - (H - 1.0) * .5) / focal_y, 1.0f};
	float dir_norm = sqrt(dir_camera[0] * dir_camera[0] + dir_camera[1] * dir_camera[1] + dir_camera[2] * dir_camera[2]);
	for (int dir_idx=0; dir_idx < 3; dir_idx++)
		dir_camera[dir_idx] = dir_camera[dir_idx] / dir_norm;
	glm::vec3 dir = {
		viewmatrix[0] * dir_camera[0] + viewmatrix[1] * dir_camera[1] + viewmatrix[2] * dir_camera[2],
		viewmatrix[4] * dir_camera[0] + viewmatrix[5] * dir_camera[1] + viewmatrix[6] * dir_camera[2],
		viewmatrix[8] * dir_camera[0] + viewmatrix[9] * dir_camera[1] + viewmatrix[10] * dir_camera[2]
	};
	float zscale = fabsf(dir_camera[2]);

	float one_over_focalx = 1. / focal_x;
	float one_over_focaly = 1. / focal_y;
	float ddir_dmean2Dx[3] = {
		viewmatrix[0] * one_over_focalx,
		viewmatrix[4] * one_over_focalx,
		viewmatrix[8] * one_over_focalx
	};
	float ddir_dmean2Dy[3] = {
		viewmatrix[1] * one_over_focaly,
		viewmatrix[5] * one_over_focaly,
		viewmatrix[9] * one_over_focaly
	};

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			if(coll_id >= 0) {
				collected_xy[block.thread_rank()] = points_xy_image[coll_id];
				for (int i = 0; i < C; i++)
					collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
				for (int i = 0; i < 3; i++)
					collected_mean3D[i + 3 * block.thread_rank()] = means3D[coll_id * 3 + i];
				for (int i = 0; i < 9; i++)
					collected_Rs[i + 9 * block.thread_rank()] = rotations[coll_id * 9 + i];
				collected_opacity[block.thread_rank()] = opacities[coll_id];
				collected_acutance[block.thread_rank()] = acutances[coll_id];
				collected_l1l2rate[block.thread_rank()] = l1l2_rates[coll_id];
				collected_op_tu[block.thread_rank()] = op_tu[coll_id];
				collected_op_tv[block.thread_rank()] = op_tv[coll_id];
				collected_op_n[block.thread_rank()] = op_n[coll_id];
				for (int i = 0; i < 3; i++)
					collected_op[i + 3 * block.thread_rank()] = op[coll_id * 3 + i];
				for (int i = 0; i < KERNEL_K; i++)
					collected_scales[i + KERNEL_K * block.thread_rank()] = scales[coll_id * KERNEL_K + i];
				for (int i = 0; i < KERNEL_K; i++)
					collected_thetas[i + KERNEL_K * block.thread_rank()] = thetas[coll_id * KERNEL_K + i];
			}
		}
		block.sync();

		// CACHE Array to sort the surfels
		int next_array[CACHE_SIZE] = {0};
		int prev_array[CACHE_SIZE] = {0};
		for(int ii=0; ii<CACHE_SIZE; ii++) {
			js[ii] = -1;
			next_array[ii] = -1;
			prev_array[ii] = -1;
		}
		int head = 0;
		int tail = 0;
		int last_cursor = 0;
		int length = 0;

		for(int jj=0; (jj < min(BLOCK_SIZE, toDo) || length > 0) && !done; jj++) {
			// Keep track of current position in range
			contributor++;

			// Check if the surfel is valid
			if(collected_id[jj] < 0) {
				continue;
			}
			
			int j = -1;
			if(cache_sort) {
				// There are unprocessed surfels
				if(jj < min(BLOCK_SIZE, toDo)) {
					// Pre-calculate Intersection (dir, cam_pos -> uv)
					const float * Rs = collected_Rs + jj * 9;
					const float * mean3D = collected_mean3D + jj * 3;
					const glm::vec3 normal = {Rs[2], Rs[5], Rs[8]};
					float dir_dot_n = Rs[2] * dir[0] + Rs[5] * dir[1] + Rs[8] * dir[2];
					float ray_t = collected_op_n[jj] / dir_dot_n;
					if (fabsf(dir_dot_n) < 0.0001f || ray_t < 0.001f || ray_t > FARTHEST_DISTANCE) {
						continue;
					}
					else{
						j = push_and_pop_array(js, ray_t_array, next_array, prev_array, head, tail, length, last_cursor, jj, ray_t);
					}
				}
				else
					j = push_and_pop_array(js, ray_t_array, next_array, prev_array, head, tail, length, last_cursor, -1, -1.f);
			}
			else
				j = jj;

			// process in cache sorted order
			if (j < 0)
				continue;
			float2 xy = collected_xy[j];
			
			float opacity = opacities[collected_id[j]];
			if (opacity < 1.0f / 255.0f)
				continue;
			
			// Step 1. Intersection (dir, cam_pos -> uv)
			const float * Rs = collected_Rs + j * 9;
			const float * mean3D = collected_mean3D + j * 3;
			const glm::vec3 normal = {Rs[2], Rs[5], Rs[8]};
			float dir_dot_n = Rs[2] * dir[0] + Rs[5] * dir[1] + Rs[8] * dir[2];
			// Truncate to avoid inconsistency
			const float float_accuracy_scale = 1e7;
	        const float dir_dot_n_abs = roundf(fabsf(dir_dot_n) * float_accuracy_scale) / float_accuracy_scale;
			dir_dot_n = copysignf(dir_dot_n_abs, dir_dot_n);
			float ray_t = collected_op_n[j] / dir_dot_n;
			if (fabsf(dir_dot_n) < 0.0001f)
				continue;
			if (ray_t < 0.001f)
				continue;
			float dir_dot_tu = dir[0] * Rs[0] + dir[1] * Rs[3] + dir[2] * Rs[6];
			float dir_dot_tv = dir[0] * Rs[1] + dir[1] * Rs[4] + dir[2] * Rs[7];
			float2 uv = {ray_t*dir_dot_tu-collected_op_tu[j], ray_t*dir_dot_tv-collected_op_tv[j]};

			// Step 2. Determine the interval
			const float * scale = collected_scales + KERNEL_K*j;
			float uv_l2norm = uv.x * uv.x + uv.y * uv.y;
			// Numerical Protection!
			uv_l2norm = max(uv_l2norm, 0.00000001);
			float theta = acos(uv.x / sqrt(uv_l2norm));
			theta = uv.y > 0? theta: 2.0f*PI - theta;
			theta = min(theta / (2.0f * PI), 1.0f);
			const float * thetas_array = collected_thetas + KERNEL_K*j;
			int k = 0;
			for(int ii=0; ii<KERNEL_K-1; ii++)
				k += thetas_array[ii]>=theta? 0: 1;
			float theta_l = k==0? 0.0f: thetas_array[k-1];
			float theta_r = thetas_array[k];
			float linear_rate = (theta - theta_l) / (theta_r - theta_l);
			float rate = 0.5f * (cosf((1.0f - linear_rate) * PI) + 1);
			float scale_left = scale[k];
			float scale_right = k==(KERNEL_K-1)? scale[0]: scale[k+1];
			
			// Step 3. Affine transformation
			float2 e1 = {cosf(theta_l * 2.0f * PI) * scale_left,  sinf(theta_l * 2.0f * PI) * scale_left};
			float2 e2 = {cosf(theta_r * 2.0f * PI) * scale_right, sinf(theta_r * 2.0f * PI) * scale_right};
			float delta = e1.x * e2.y - e1.y * e2.x;
			delta = copysignf(max(fabsf(delta), 0.0000001f), delta);
			float2 uv_t = {(e2.y * uv.x - e2.x * uv.y) / delta, (- e1.y * uv.x + e1.x * uv.y) / delta};

			// Step 4. Kernel function (theta, uv_norm -> kernel_opacity)
			float l1l2_rate = collected_l1l2rate[j];
			float uvt_l1norm = fabsf(uv_t.x) + fabsf(uv_t.y);
			uvt_l1norm = uvt_l1norm * uvt_l1norm * 0.5f;
			uvt_l1norm = max(uvt_l1norm, 0.00000001);
			float uvt_l2norm = uv_l2norm * (rate / (scale_right*scale_right) + (1 - rate) / (scale_left*scale_left)) * 0.5f;
			uvt_l2norm = max(uvt_l2norm, 0.00000001);
			float uv_norm = l1l2_rate * uvt_l1norm + (1 - l1l2_rate) * uvt_l2norm;
			float kernel_opacity = exp(- uv_norm);

			// Step 5. Sharpening
			const float k_acu = collected_acutance[j];
			float sharpen = 0.f;
			float alpha   = 0.f;
			float old_alpha = 0.f;
			bool cond1 = false;
			bool cond2 = false;
			if(SHARPEN_ALPHA){
				alpha = kernel_opacity * opacity;
				cond1 = (alpha < ((1.0f + k_acu) / 4.0f));
				cond2 = (alpha < ((3.0f - k_acu) / 4.0f));
				sharpen = cond1? ((1.0f - k_acu) / (1.0f + k_acu) * alpha): ((1.0f + k_acu) / (1.0f - k_acu) * alpha - k_acu / (1.0f - k_acu));
				sharpen = cond2? sharpen: ((1.0f - k_acu) / (1.0f + k_acu) * alpha + 2.0f * k_acu / (1.0f + k_acu));
				old_alpha = alpha;
				alpha = sharpen;
			}
			else{
				cond1 = (kernel_opacity < ((1.0f + k_acu) / 4.0f));
				cond2 = (kernel_opacity < ((3.0f - k_acu) / 4.0f));
				sharpen = cond1? ((1.0f - k_acu) / (1.0f + k_acu) * kernel_opacity): ((1.0f + k_acu) / (1.0f - k_acu) * kernel_opacity - k_acu / (1.0f - k_acu));
				sharpen = cond2? sharpen: ((1.0f - k_acu) / (1.0f + k_acu) * kernel_opacity + 2.0f * k_acu / (1.0f + k_acu));
				alpha = sharpen * opacity;
			}

			// Step 6. Low pass filter
			float gs_dir[3] = {mean3D[0]-cam_pos[0], mean3D[1]-cam_pos[1], mean3D[2]-cam_pos[2]};
			float gs_dir_norm = sqrtf(max(gs_dir[0] * gs_dir[0] + gs_dir[1] * gs_dir[1] + gs_dir[2] * gs_dir[2], 0.0000001f));
			for(int ii=0; ii<3; ii++)
				gs_dir[ii] /= gs_dir_norm;
			float gsdir_dot_n = gs_dir[0] * normal[0] + gs_dir[1] * normal[1] + gs_dir[2] * normal[2];
			float cos_dn = fabsf(gsdir_dot_n);
			float2 res_2d = {(xy.x - pixf.x) / cos_dn, (xy.y - pixf.y) / cos_dn};
			float dist_2d_norm = FilterInvSquare * (res_2d.x * res_2d.x + res_2d.y * res_2d.y);
			float G_lps = exp(-0.5f * dist_2d_norm);
			float alpha_lpf = opacity * G_lps;
			bool filter_cond = (alpha < alpha_lpf) & LOW_PASS_FILTER;
			alpha = filter_cond? alpha_lpf: alpha;
			float pix_depth = filter_cond? depths[collected_id[j]]: ray_t * zscale;

			if (alpha < 1.0f / 255.0f)
				continue;

			// Step 5. Alpha blending
			alpha = min(0.99f, alpha);
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)  // Only when T < 0.01, since alpha <= 0.99
			{
				done = true;
				continue;
			}
			for (int ch = 0; ch < C; ch++)
				Ci[ch] += collected_colors[ch * BLOCK_SIZE + j] * alpha * T;
			Di += pix_depth * alpha * T;
			for(int ch=0; ch < 3; ch++)
				Ni[ch] += normal[ch] * copysignf(1., -dir_dot_n) * alpha * T;
			float old_T = T;
			T = test_T;

			// #######################################################################
			// Backward from output to each pixel to COLOR, ALPHA, NORMAL
			// #######################################################################
			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			const float dchannel_dcolor = alpha * old_T;
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (old_T * collected_colors[ch * BLOCK_SIZE + j] - (color_final[ch] - Ci[ch] + T_final * bg_color[ch]) / (1 - alpha)) * dL_dchannel;
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			for(int ch=0; ch < 3; ch ++) {
				const float n = normal[ch] * copysignf(1., -dir_dot_n);
				const float dchannel_dnormal = dchannel_dcolor * copysignf(1., -dir_dot_n);
				const float dL_dchannel = dL_dpixel_normal[ch];
				dL_dalpha += (old_T * n - (normal_final[ch] - Ni[ch]) / (1 - alpha)) * dL_dchannel;
				atomicAdd(&(dL_drotations[global_id*9+ch*3+2]), dchannel_dnormal * dL_dchannel);
			}
			dL_dalpha += (old_T - (T_final - old_T) / (1 - alpha)) * dL_dpixel_alpha;
			dL_dalpha += (old_T * pix_depth - (depth_final - Di) / (1 - alpha)) * dL_ddepth;

			if(filter_cond){
				// // Half the gradient
				// dL_dalpha *= 0.5f;
				// Backward for mean3D
				float dL_dscreenx = dL_dalpha * (- alpha_lpf * FilterInvSquare * res_2d.x / cos_dn);
				float dL_dscreeny = dL_dalpha * (- alpha_lpf * FilterInvSquare * res_2d.y / cos_dn);
				float dscreenx_dcamerax = focal_x / pix_depth;
				float dscreeny_dcameray = focal_y / pix_depth;
				float dcamerax_dmean3D[3] = {viewmatrix[0], viewmatrix[1], viewmatrix[2]};
				float dcameray_dmean3D[3] = {viewmatrix[4], viewmatrix[5], viewmatrix[6]};
				float dcameraz_dmean3D[3] = {viewmatrix[8], viewmatrix[9], viewmatrix[10]};
				atomicAdd(&dL_dmean2D[global_id].x, dL_dscreenx * ddelx_dx);
				atomicAdd(&dL_dmean2D[global_id].y, dL_dscreeny * ddely_dy);
				atomicAdd(&dL_dmean2D_densify[global_id].x, fabsf(dL_dscreenx * ddelx_dx));
				atomicAdd(&dL_dmean2D_densify[global_id].y, fabsf(dL_dscreeny * ddely_dy));
				for(int ii=0; ii<3; ii++){
					float dL_dmean3D_ii = dL_dscreenx*dscreenx_dcamerax*dcamerax_dmean3D[ii] + dL_dscreeny*dscreeny_dcameray*dcameray_dmean3D[ii];
					atomicAdd(&(dL_dmeans3D[global_id*3+ii]), dL_dmean3D_ii);
				}
				// Backward for opacity
				float dalpha_dopacity = G_lps;
				atomicAdd(&(dL_dopacity[global_id]), dL_dalpha * dalpha_dopacity);
				atomicAdd(&dL_dopacity_densify[global_id], fabsf(dL_dalpha * dalpha_dopacity));
				// Backward for normal
				float dalpha_dcos = FilterInvSquare * alpha_lpf / cos_dn * (res_2d.x * res_2d.x + res_2d.y * res_2d.y);
				float dcos_ddot = copysignf(1.0f, gsdir_dot_n);
				atomicAdd(&(dL_drotations[global_id*9+2]), dL_dalpha * dalpha_dcos * dcos_ddot * gs_dir[0]);
				atomicAdd(&(dL_drotations[global_id*9+5]), dL_dalpha * dalpha_dcos * dcos_ddot * gs_dir[1]);
				atomicAdd(&(dL_drotations[global_id*9+8]), dL_dalpha * dalpha_dcos * dcos_ddot * gs_dir[2]);
			}
			else {
				////////////////////////////////////////////////////////////////////
				// UV Backward for mean3D
				float dtu_over_dn = dir_dot_tu / dir_dot_n;
				float du_dmean3D[3] = {Rs[2] * dtu_over_dn - Rs[0], Rs[5] * dtu_over_dn - Rs[3], Rs[8] * dtu_over_dn - Rs[6]};
				float dtv_over_dn = dir_dot_tv / dir_dot_n;
				float dv_dmean3D[3] = {Rs[2] * dtv_over_dn - Rs[1], Rs[5] * dtv_over_dn - Rs[4], Rs[8] * dtv_over_dn - Rs[7]};
				// UV Backward for R
				float tangent[3] = {ray_t*dir[0] - op[collected_id[j]*3], ray_t*dir[1] - op[collected_id[j]*3+1], ray_t*dir[2] - op[collected_id[j]*3+2]};
				float du_dR[9] = {0.f};
				float dv_dR[9] = {0.f};
				for(int ii=0; ii < 3; ii++) {
					du_dR[3 * ii] = tangent[ii];
					dv_dR[3 * ii + 1] = tangent[ii];
				}
				for(int ii=0; ii<3; ii++) {
					du_dR[3*ii+2] = - dtu_over_dn * tangent[ii];
					dv_dR[3*ii+2] = - dtv_over_dn * tangent[ii];
				}
				// rayt Backward for depth (ray_t)
				float drayt_dn[3] = {
					(op[collected_id[j]*3]   * dir_dot_n - op_n[collected_id[j]] * dir[0]) / dir_dot_n / dir_dot_n,
					(op[collected_id[j]*3+1] * dir_dot_n - op_n[collected_id[j]] * dir[1]) / dir_dot_n / dir_dot_n,
					(op[collected_id[j]*3+2] * dir_dot_n - op_n[collected_id[j]] * dir[2]) / dir_dot_n / dir_dot_n
				};
				float drayt_dmean3D[3] = {normal[0] / dir_dot_n, normal[1] / dir_dot_n, normal[2] / dir_dot_n};
				// Backward for UV_T w.r.t UV, theta, scale
				float dut_du =  e2.y / delta;
				float dvt_du = -e1.y / delta;
				float dut_dv = -e2.x / delta;
				float dvt_dv =  e1.x / delta;
				float sin2 = sinf(2*PI*(theta_r-theta_l));
				float cos2 = cosf(2*PI*(theta_r-theta_l));
				float twopi_over_tan = 2 * PI * cos2 / sin2;
				float dut_dtheta_left  =   uv_t.x * twopi_over_tan;
				float dvt_dtheta_left  = - 2 * PI * scale_left * uv_t.x / (scale_right * sin2);
				float dut_dtheta_right =   2 * PI * scale_right * uv_t.y / (scale_left * sin2);
				float dvt_dtheta_right = - uv_t.y * twopi_over_tan;
				float dut_dscale_left  = - uv_t.x / scale_left;
				float dvt_dscale_left  = 0.f;
				float dut_dscale_right = 0.f;
				float dvt_dscale_right = - uv_t.y / scale_right;
				float duvtl1_dut = copysignf(uv_t.y, uv_t.x) + uv_t.x;
				float duvtl1_dvt = copysignf(uv_t.x, uv_t.y) + uv_t.y;
				// Backward for uvt_l1norm w.r.t UV, theta, scale
				float duvtl1_du = duvtl1_dut * dut_du + duvtl1_dvt * dvt_du;
				float duvtl1_dv = duvtl1_dut * dut_dv + duvtl1_dvt * dvt_dv;
				float duvtl1_dtheta_left  = duvtl1_dut * dut_dtheta_left  + duvtl1_dvt * dvt_dtheta_left;
				float duvtl1_dtheta_right = duvtl1_dut * dut_dtheta_right + duvtl1_dvt * dvt_dtheta_right;
				float duvtl1_dscale_left  = duvtl1_dut * dut_dscale_left  + duvtl1_dvt * dvt_dscale_left;
				float duvtl1_dscale_right = duvtl1_dut * dut_dscale_right + duvtl1_dvt * dvt_dscale_right;
				// Backward for rate w.r.t UV, thetas
				float one_over_thetarml = 1 / (theta_r - theta_l);
				float drate_dlinear = .5 * PI * sinf(PI * linear_rate);
				float drate_dtheta = one_over_thetarml * drate_dlinear;
				float drate_du = -uv.y / uv_l2norm / (2.0f * PI) * drate_dtheta;
				float drate_dv =  uv.x / uv_l2norm / (2.0f * PI) * drate_dtheta;
				float drate_dthetas_left  = (theta - theta_r) * one_over_thetarml * one_over_thetarml * drate_dlinear;
				float drate_dthetas_right = (theta_l - theta) * one_over_thetarml * one_over_thetarml * drate_dlinear;
				// Backward for uvt_l2norm w.r.t UV, theta, scale
				float duvtl2_drate = 0.5f * uv_l2norm * (1 / (scale_right*scale_right) - 1 / (scale_left*scale_left));
				float rated_scale = rate / (scale_right*scale_right) + (1-rate) / (scale_left*scale_left);
				float duvtl2_du = uv.x * rated_scale + duvtl2_drate * drate_du;
				float duvtl2_dv = uv.y * rated_scale + duvtl2_drate * drate_dv;
				float duvtl2_dstheta_left  = duvtl2_drate * drate_dthetas_left;
				float duvtl2_dstheta_right = duvtl2_drate * drate_dthetas_right;
				float duvtl2_dscale_left  = - uv_l2norm * (1 - rate) / (scale_left *scale_left *scale_left);
				float duvtl2_dscale_right = - uv_l2norm * rate       / (scale_right*scale_right*scale_right);
				// Backward for uvt_norm w.r.t UV
				float duvnorm_duvtl1 = l1l2_rate;
				float duvnorm_duvtl2 = (1 - l1l2_rate);
				float duvnorm_du = duvnorm_duvtl1 * duvtl1_du + duvnorm_duvtl2 * duvtl2_du;
				float duvnorm_dv = duvnorm_duvtl1 * duvtl1_dv + duvnorm_duvtl2 * duvtl2_dv;
				// Backward for uvt_norm w.r.t mean3D, R, theta, scale, l1l2rate
				float duvnorm_dmean3D[3] = {
					duvnorm_du * du_dmean3D[0] + duvnorm_dv * dv_dmean3D[0],
					duvnorm_du * du_dmean3D[1] + duvnorm_dv * dv_dmean3D[1],
					duvnorm_du * du_dmean3D[2] + duvnorm_dv * dv_dmean3D[2],
				};
				float duvnorm_dR[9] = {0.f};
				for(int ii=0; ii<9; ii++)
					duvnorm_dR[ii] = duvnorm_du * du_dR[ii] + duvnorm_dv * dv_dR[ii];
				float duvnorm_dtheta_left  = duvnorm_duvtl1 * duvtl1_dtheta_left  + duvnorm_duvtl2 * duvtl2_dstheta_left;
				float duvnorm_dtheta_right = duvnorm_duvtl1 * duvtl1_dtheta_right + duvnorm_duvtl2 * duvtl2_dstheta_right;
				float duvnorm_dscale_left  = duvnorm_duvtl1 * duvtl1_dscale_left  + duvnorm_duvtl2 * duvtl2_dscale_left;
				float duvnorm_dscale_right = duvnorm_duvtl1 * duvtl1_dscale_right + duvnorm_duvtl2 * duvtl2_dscale_right;
				float duvnorm_dl1l2rate = uvt_l1norm - uvt_l2norm;
				// Kernel Opacity Backward for scale and thetas
				int idx_scale_left = k;
				int idx_scale_right = k==(KERNEL_K-1)? 0: k+1;
				int idx_theta_left = k-1; // check >= 0
				int idx_theta_right = k;
				// Result
				float dko_duvnorm       = - kernel_opacity;
				float dko_dscale_right  = dko_duvnorm * duvnorm_dscale_right;
				float dko_dscale_left   = dko_duvnorm * duvnorm_dscale_left;
				float dko_dthetas_left  = dko_duvnorm * duvnorm_dtheta_left;
				float dko_dthetas_right = dko_duvnorm * duvnorm_dtheta_right;
				const float ddepth_drayt    = dchannel_dcolor * zscale;
				////////////////////////////////////////////////////////////////////

				// Backward for alpha w.r.t ko, opacity, k_acu
				float dalpha_dko      = 0.f;
				float dalpha_dopacity = 0.f;
				float dalpha_dk_acu   = 0.f;
				if(SHARPEN_ALPHA){
					const float doldalpha_dko      = opacity;
					const float doldalpha_dopacity = kernel_opacity;
					float dsharpen_dk_acu = cond1? (- 2.0f * old_alpha / (1.0f + k_acu) / (1.0f + k_acu)): ((2.0f * old_alpha - 1.0f) / (1.0f - k_acu) / (1.0f - k_acu));
					dsharpen_dk_acu = cond2? dsharpen_dk_acu: (- 2.0f * (old_alpha - 1.0f) / (1.0f + k_acu) / (1.0f + k_acu));
					float dsharpen_doldalpha = cond1? ((1.0f - k_acu) / (1.0f + k_acu)): ((1.0f + k_acu) / (1.0f - k_acu));
					dsharpen_doldalpha = cond2? dsharpen_doldalpha: (1.0f - k_acu) / (1.0f + k_acu);
					// alpha = sharpen(ko*opacity, k_acu)
					dalpha_dko      = dsharpen_doldalpha * doldalpha_dko;
					dalpha_dopacity = dsharpen_doldalpha * doldalpha_dopacity;
					dalpha_dk_acu   = dsharpen_dk_acu;
				}
				else{
					const float dalpha_dsharpen = opacity;
					float dsharpen_dk_acu = cond1? (- 2.0f * kernel_opacity / (1.0f + k_acu) / (1.0f + k_acu)): ((2.0f * kernel_opacity - 1.0f) / (1.0f - k_acu) / (1.0f - k_acu));
					dsharpen_dk_acu = cond2? dsharpen_dk_acu: (- 2.0f * (kernel_opacity - 1.0f) / (1.0f + k_acu) / (1.0f + k_acu));
					float dsharpen_dko = cond1? ((1.0f - k_acu) / (1.0f + k_acu)): ((1.0f + k_acu) / (1.0f - k_acu));
					dsharpen_dko = cond2? dsharpen_dko: (1.0f - k_acu) / (1.0f + k_acu);
					// alpha = sharpen(ko, k_acu) * opacity
					dalpha_dko      = dalpha_dsharpen * dsharpen_dko;
					dalpha_dopacity = sharpen;
					dalpha_dk_acu   = dalpha_dsharpen * dsharpen_dk_acu;
				}

				float dL_dk_acu = dL_dalpha * dalpha_dk_acu;
				const float dL_dko    = dL_dalpha * dalpha_dko;
				const float dko_du = dko_duvnorm * duvnorm_du;
				const float dko_dv = dko_duvnorm * duvnorm_dv;

				// Update gradients w.r.t. opacity of the Gaussian
				atomicAdd(&(dL_dopacity[global_id]), dL_dalpha * dalpha_dopacity);
				atomicAdd(&dL_dopacity_densify[global_id], fabsf(dL_dalpha * dalpha_dopacity));
				// Update gradients w.r.t. acutance of the Gaussian
				atomicAdd(&(dL_dacutance[global_id]), dL_dk_acu);

				// Update gradients w.r.t. scales of the Gaussian
				atomicAdd(&(dL_dscale[global_id*KERNEL_K+idx_scale_left]),  dL_dko*dko_dscale_left);
				atomicAdd(&(dL_dscale[global_id*KERNEL_K+idx_scale_right]), dL_dko*dko_dscale_right);
				// Update gradients w.r.t. thetas
				if (idx_theta_left >= 0)
					atomicAdd(&(dL_dthetas[global_id*KERNEL_K+idx_theta_left]), dL_dko*dko_dthetas_left);
				atomicAdd(&(dL_dthetas[global_id*KERNEL_K+idx_theta_right]), dL_dko*dko_dthetas_right);
				float dL_dmean2D_tmp[2] = {0.f};
				// Update gradients w.r.t. means3D
				for(int ii=0; ii<3; ii++) {
					float dL_dmean3D_ii = dL_dko*dko_duvnorm*duvnorm_dmean3D[ii];
					atomicAdd(&(dL_dmeans3D[global_id*3+ii]), dL_dmean3D_ii);
					// Update gradients w.r.t. 2D mean position of the Gaussian
					float dL_dmean2Dx_from_3Dii = dL_dmean3D_ii * ray_t * zscale * ddir_dmean2Dx[ii] * ddelx_dx;
					float dL_dmean2Dy_from_3Dii = dL_dmean3D_ii * ray_t * zscale * ddir_dmean2Dy[ii] * ddely_dy;
					dL_dmean2D_tmp[0] += dL_dmean2Dx_from_3Dii;
					dL_dmean2D_tmp[1] += dL_dmean2Dy_from_3Dii;
					// Update gradients for densification
					atomicAdd(&dL_dmean2D[global_id].x, dL_dmean2Dx_from_3Dii);
					atomicAdd(&dL_dmean2D[global_id].y, dL_dmean2Dy_from_3Dii);
				}
				// Update gradients for densification
				atomicAdd(&dL_dmean2D_densify[global_id].x, fabsf(dL_dmean2D_tmp[0]));
				atomicAdd(&dL_dmean2D_densify[global_id].y, fabsf(dL_dmean2D_tmp[1]));
				// Update gradients w.r.t. rotation
				for(int ii=0; ii<9; ii++)
					atomicAdd(&(dL_drotations[global_id*9+ii]), dL_dko*dko_duvnorm*duvnorm_dR[ii]);
				// Update gradients w.r.t. l1l2_rates
				atomicAdd(&(dL_dl1l2_rates[global_id]), dL_dko*dko_duvnorm*duvnorm_dl1l2rate);
				// Update depth gradients w.r.t normal and mean3D
				float dL_drayt = dL_ddepth * ddepth_drayt;
				for(int ii=0; ii<3; ii++) {
					atomicAdd(&(dL_dmeans3D[global_id*3+ii]), dL_drayt*drayt_dmean3D[ii]);
					atomicAdd(&(dL_drotations[global_id*9+ii*3+2]), dL_drayt*drayt_dn[ii]);
				}
			}
		}
	}
}


void BACKWARD::render(
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

	bool cache_sort)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		colors,
		depths,

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

		op,
		op_tu,
		op_tv,
		op_n,

		final_Ts,
		final_Colors,
		final_Depths,
		final_Normals,
	
		n_contrib,
		dL_dpixels,
		dL_dpixels_alpha,
		dL_ddepths,
		dL_dnormal,
		dL_dmean2D,
		dL_dmean2D_densify,
		dL_dopacity_densify,
		dL_dopacity,
		dL_dcolors,
		dL_dacutance,

		dL_dmeans3D,
		dL_drotations,
		dL_dscale,
		dL_dthetas,
		dL_dl1l2_rates,

		cache_sort
		);
}


template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* campos,
	float* dL_dcolor,
	float* dL_dsh,
	float* dL_dmeans3D)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans3D, (glm::vec3*)dL_dsh);
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* campos,
	float* dL_dcolor,
	float* dL_dsh,
	float* dL_dmeans3D)
{

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		campos,
		dL_dcolor,
		dL_dsh,
		dL_dmeans3D);
}
