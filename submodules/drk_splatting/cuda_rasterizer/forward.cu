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

#include <math.h>
#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#if DRK_FORCE_ACUTANCE_ONE_KERNEL && DRK_FORCE_L1_KERNEL
#define DRK_ACU1_L1_ANGLELESS_KERNEL 1
#else
#define DRK_ACU1_L1_ANGLELESS_KERNEL 0
#endif

#if KERNEL_K <= 4 || ((DRK_ACU1_L1_ANGLELESS_KERNEL || DRK_SCREEN_SPACE_L1_KERNEL) && KERNEL_K <= 8)
#define DRK_SHARE_KERNEL_GEOMETRY 1
#else
#define DRK_SHARE_KERNEL_GEOMETRY 0
#endif

__device__ __constant__ float pi = 3.14159265358979323846f;
constexpr float DRK_ACUTANCE_ONE_EPS = 1.0e-6f;
constexpr float DRK_LN2 = 0.6931471805599453f;

__forceinline__ __device__ bool drk_angle_upper_half(const float2 v) {
	return (v.y > 0.0f) || (v.y == 0.0f && v.x >= 0.0f);
}

__forceinline__ __device__ bool drk_angle_less_0_2pi(const float2 a, const float2 b) {
	const bool a_upper = drk_angle_upper_half(a);
	const bool b_upper = drk_angle_upper_half(b);
	if (a_upper != b_upper)
		return a_upper && !b_upper;
	return (a.x * b.y - a.y * b.x) > 0.0f;
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for
	// "Differentiable Point-Based Radiance Fields for
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];

				if (deg > 3)
				{
					result = result +
						SH_C4[0] * xy * (xx - yy) * sh[16] +
						SH_C4[1] * yz * (3.0f * xx - yy) * sh[17] +
						SH_C4[2] * xy * (7.0f * zz - 1.0f) * sh[18] +
						SH_C4[3] * yz * (7.0f * zz - 3.0f) * sh[19] +
						SH_C4[4] * (zz * (35.0f * zz - 30.0f) + 3.0f) * sh[20] +
						SH_C4[5] * xz * (7.0f * zz - 3.0f) * sh[21] +
						SH_C4[6] * (xx - yy) * (7.0f * zz - 1.0f) * sh[22] +
						SH_C4[7] * xz * (xx - 3.0f * yy) * sh[23] +
						SH_C4[8] * (xx * (xx - 3.0f * yy) - yy * (3.0f * xx - yy)) * sh[24];
				}
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float computeRadius2D(const float3& mean, const float* scale, float scale_modifier, const glm::vec3 normal, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* viewmatrix)
{
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	const float focal = max(focal_x, focal_y);
	float max_scale = 0.0f;
	for (int i=0; i<KERNEL_K; i++) {
		float scale_i = scale[i] * scale_modifier;
		if (scale_i > max_scale)
			max_scale = scale_i;
	}
	const float radius = max_scale / fabsf(t.z) * focal;

	return radius;
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void Q2R(const glm::vec4 rot, float * R)
{
	float r = rot.x;
	float x = rot.y;
	float y = rot.z;
	float z = rot.w;

	R[0] = 1.f - 2.f * (y * y + z * z);
	R[1] = 2.f * (x * y - r * z);
	R[2] = 2.f * (x * z + r * y);
	R[3] = 2.f * (x * y + r * z);
	R[4] = 1.f - 2.f * (x * x + z * z);
	R[5] = 2.f * (y * z - r * x);
	R[6] = 2.f * (x * z - r * y);
	R[7] = 2.f * (y * z + r * x);
	R[8] = 1.f - 2.f * (x * x + y * y);
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocess2DCUDA(int P, int D, int M,
	const float* means3D,
	const float* scales,
	const float* thetas,
	const float* opacities,
	const float* acutances,
	const float scale_modifier,
	const float* Rs,
	const float* shs,
	bool* clamped,
	const float* colors_precomp,
		const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
		const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* rgb,
	const dim3 grid,
	uint32_t* tiles_touched,

	float * op,
	float * op_tu,
	float * op_tv,
	float * op_n,
	float * op_cos_n,
	float * op_inv_cos_n,
		float2 * kernel_vecs,
		float * scale_inv2,
		float * kernel_inv_delta,
		uint2* stored_rect_min,
		uint2* stored_rect_max,

		bool prefiltered,
	bool tile_culling,
	bool store_backward_aux
)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	float opacity = opacities[idx];
	if (opacity < 1.0f / 255.0f)
		return;

	// // Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, means3D, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	scales = scales + idx * KERNEL_K;
	thetas = thetas + idx * KERNEL_K;
	Rs = Rs + idx * 9;
	float2 local_kernel_vecs[KERNEL_K];
	for(int ii=0; ii<KERNEL_K; ii++) {
		float theta = ii==0? 0: thetas[ii-1];
		const float angle = theta * 2.f * PI;
		float s, c;
		sincosf(angle, &s, &c);
		local_kernel_vecs[ii] = {c * scales[ii], s * scales[ii]};
#if !DRK_SCREEN_SPACE_L1_KERNEL
		kernel_vecs[idx * KERNEL_K + ii] = local_kernel_vecs[ii];
#endif
#if !DRK_FORCE_L1_KERNEL
		scale_inv2[idx * KERNEL_K + ii] = __frcp_rn(scales[ii] * scales[ii]);
#endif
	}
	for(int ii=0; ii<KERNEL_K; ii++) {
		float2 e1 = local_kernel_vecs[ii];
		float2 e2 = local_kernel_vecs[ii==(KERNEL_K-1)? 0: ii+1];
		float delta = e1.x * e2.y - e1.y * e2.x;
		delta = copysignf(fmaxf(fabsf(delta), 0.0000001f), delta);
		kernel_inv_delta[idx * KERNEL_K + ii] = __frcp_rn(delta);
	}

	// Transform point by projecting
	float3 p_orig = { means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// Compute 2D screen-space radius
	glm::vec3 normal = {Rs[2], Rs[5], Rs[8]};
	float my_radius = computeRadius2D(p_orig, scales, scale_modifier, normal, focal_x, focal_y, tan_fovx, tan_fovy, viewmatrix);
	opacity = max(opacity, 0.001f);

	float inv_sharpen = 0.f;
#if DRK_FORCE_ACUTANCE_ONE_KERNEL
	inv_sharpen = 0.5f;
#else
	if(SHARPEN_ALPHA){
		float k_acu = min(0.9f, acutances[idx]);
		float threshold_opacity = exp(- 3.0f * 3.0f / 2.0f);
		inv_sharpen = (1.0f + k_acu) / (1.0f - k_acu) * threshold_opacity;
		inv_sharpen       = (threshold_opacity < ((1.0f - k_acu) / 4.0f))? inv_sharpen: (1.0f - k_acu) / (1.0f + k_acu) * threshold_opacity + k_acu / (1.0f + k_acu);
		inv_sharpen       = (threshold_opacity < ((3.0f + k_acu) / 4.0f))? inv_sharpen: (1.0f + k_acu) / (1.0f - k_acu) * threshold_opacity - 2.0f * k_acu / (1.0f - k_acu);
		inv_sharpen = max(min(inv_sharpen / opacity, 0.999f), 0.001f);
	}
	else{
		float k_acu_raw = acutances[idx];
		if (k_acu_raw >= 1.0f - DRK_ACUTANCE_ONE_EPS) {
			// Sharpness 1 is the limiting hard boundary: kernel opacity >= 0.5.
			inv_sharpen = 0.5f;
		}
		else {
			float k_acu = min(0.999999f, k_acu_raw);
			float threshold_opacity = min(exp(- 3.0f * 3.0f / 2.0f), (1.0f - k_acu) / 8.0f);
			inv_sharpen = (1.0f + k_acu) / (1.0f - k_acu) * threshold_opacity;
			inv_sharpen       = (threshold_opacity < ((1.0f - k_acu) / 4.0f))? inv_sharpen: (1.0f - k_acu) / (1.0f + k_acu) * threshold_opacity + k_acu / (1.0f + k_acu);
			inv_sharpen       = (threshold_opacity < ((3.0f + k_acu) / 4.0f))? inv_sharpen: (1.0f + k_acu) / (1.0f - k_acu) * threshold_opacity - 2.0f * k_acu / (1.0f - k_acu);
			inv_sharpen = max(min(inv_sharpen, 0.999999999f), 0.000000001f);
		}
	}
#endif
	float scale_ratio = sqrtf(- 2.0f * logf(inv_sharpen));
	my_radius = scale_ratio * my_radius;
	// Low pass filter
	my_radius = max(my_radius, sqrtf(9.0f / FilterInvSquare));
	my_radius = ceil(my_radius);

	float2 point_image = { ndc2PixX(p_proj.x, W), ndc2PixY(p_proj.y, H) };
	uint2 rect_min, rect_max;
#if DRK_SCREEN_SPACE_L1_KERNEL
		bool have_tight_bounds = false;
		const float boundary_scale = sqrtf(2.0f * DRK_LN2);
	bool screen_polygon_valid = true;
	for(int ii=0; ii<KERNEL_K; ii++) {
		float u = boundary_scale * local_kernel_vecs[ii].x;
		float v = boundary_scale * local_kernel_vecs[ii].y;
		float3 vert_world = {
			p_orig.x + u * Rs[0] + v * Rs[1],
			p_orig.y + u * Rs[3] + v * Rs[4],
			p_orig.z + u * Rs[6] + v * Rs[7]
		};
		float3 vert_view = transformPoint4x3(vert_world, viewmatrix);
		screen_polygon_valid &= vert_view.z > 0.01f;
		float4 vert_hom = transformPoint4x4(vert_world, projmatrix);
		float vert_w = 1.0f / (vert_hom.w + 0.0000001f);
		float3 vert_proj = {vert_hom.x * vert_w, vert_hom.y * vert_w, vert_hom.z * vert_w};
		kernel_vecs[idx * KERNEL_K + ii] = {ndc2PixX(vert_proj.x, W), ndc2PixY(vert_proj.y, H)};
	}
	if (!screen_polygon_valid)
		return;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
#elif DRK_TIGHT_EXACT_AABB && DRK_ACU1_L1_ANGLELESS_KERNEL && !LOW_PASS_FILTER
	bool have_tight_bounds = false;
	const float boundary_scale = sqrtf(2.0f * DRK_LN2);
	bool tight_rect_valid = true;
	float tight_min_x = point_image.x;
	float tight_max_x = point_image.x;
	float tight_min_y = point_image.y;
	float tight_max_y = point_image.y;
	float3 vert_center_view = transformPoint4x3(p_orig, viewmatrix);
	for(int ii=0; ii<KERNEL_K; ii++) {
		const float u = boundary_scale * local_kernel_vecs[ii].x;
		const float v = boundary_scale * local_kernel_vecs[ii].y;
		float3 vert_world = {
			p_orig.x + u * Rs[0] + v * Rs[1],
			p_orig.y + u * Rs[3] + v * Rs[4],
			p_orig.z + u * Rs[6] + v * Rs[7]
		};
		float3 vert_view = transformPoint4x3(vert_world, viewmatrix);
		tight_rect_valid &= vert_view.z > 0.01f;
		adjustVertView(vert_center_view, vert_view);
		const float safe_z = copysignf(0.0000001f + fabsf(vert_view.z), vert_view.z);
		const float2 vert_pix = {
			vert_view.x / safe_z * focal_x + drkProjectionCenterX((float)W),
			vert_view.y / safe_z * focal_y + drkProjectionCenterY((float)H)
		};
		tight_min_x = fminf(tight_min_x, vert_pix.x);
		tight_max_x = fmaxf(tight_max_x, vert_pix.x);
		tight_min_y = fminf(tight_min_y, vert_pix.y);
		tight_max_y = fmaxf(tight_max_y, vert_pix.y);
	}
	if (tight_rect_valid) {
		const float margin = DRK_TIGHT_EXACT_AABB_MARGIN;
		const float tight_center_x = 0.5f * (tight_min_x + tight_max_x);
		const float tight_center_y = 0.5f * (tight_min_y + tight_max_y);
		const float tight_half_extent = 0.5f * fmaxf(tight_max_x - tight_min_x, tight_max_y - tight_min_y);
		point_image = {tight_center_x, tight_center_y};
		my_radius = ceilf(tight_half_extent + margin);
		getRectFromBounds(
			tight_min_x - margin,
			tight_min_y - margin,
			tight_max_x + margin,
			tight_max_y + margin,
			rect_min,
			rect_max,
			grid);
		have_tight_bounds = true;
	}
	if (!have_tight_bounds)
		getRect(point_image, my_radius, rect_min, rect_max, grid);
#elif DRK_TIGHT_GENERAL_AABB
	// Tight projected-polygon AABB for the general (soft / low-pass) kernel.
	// The default circumscribing-circle radius (getRect) over-covers thin or
	// elongated K-gons by a large factor. We instead bound the K support-boundary
	// vertices (at scale_ratio, the same support radius used for my_radius) and
	// pad by a low-pass margin. This only removes empty tiles (kernel ~ 0 there),
	// so the rendered result and gradients are preserved.
	{
		bool tight_ok = true;
		float tmnx = point_image.x, tmxx = point_image.x;
		float tmny = point_image.y, tmxy = point_image.y;
		float3 vc_view = transformPoint4x3(p_orig, viewmatrix);
		// scale_ratio reaches the inv_sharpen (~0.011) opacity contour; the render
		// loop keeps contributions down to ~1/255, i.e. a larger uv_norm. Extend the
		// bounding vertices to cover that true support so no contributing tile is dropped.
		const float cover_ratio = scale_ratio * 1.18f;
		for (int ii = 0; ii < KERNEL_K; ii++) {
			const float u = cover_ratio * local_kernel_vecs[ii].x;
			const float v = cover_ratio * local_kernel_vecs[ii].y;
			float3 vw = { p_orig.x + u * Rs[0] + v * Rs[1],
			              p_orig.y + u * Rs[3] + v * Rs[4],
			              p_orig.z + u * Rs[6] + v * Rs[7] };
			float3 vv = transformPoint4x3(vw, viewmatrix);
			tight_ok &= vv.z > 0.01f;
			adjustVertView(vc_view, vv);
			const float sz = copysignf(0.0000001f + fabsf(vv.z), vv.z);
			const float px = vv.x / sz * focal_x + drkProjectionCenterX((float)W);
			const float py = vv.y / sz * focal_y + drkProjectionCenterY((float)H);
			tmnx = fminf(tmnx, px); tmxx = fmaxf(tmxx, px);
			tmny = fminf(tmny, py); tmxy = fmaxf(tmxy, py);
		}
		if (tight_ok) {
			const float margin = fmaxf((float)DRK_TIGHT_EXACT_AABB_MARGIN, sqrtf(9.0f / FilterInvSquare)) + 2.0f;
			getRectFromBounds(tmnx - margin, tmny - margin, tmxx + margin, tmxy + margin, rect_min, rect_max, grid);
			my_radius = ceilf(fmaxf(0.5f * (tmxx - tmnx), 0.5f * (tmxy - tmny)) + margin);
		} else {
			getRect(point_image, my_radius, rect_min, rect_max, grid);
		}
	}
#else
	getRect(point_image, my_radius, rect_min, rect_max, grid);
#endif
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 cam_pos_vec = {cam_pos[0], cam_pos[1], cam_pos[2]};
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)means3D, cam_pos_vec, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	stored_rect_min[idx] = rect_min;
	stored_rect_max[idx] = rect_max;

	// Project vertices of the kernel onto the image plane
	float vert2d[KERNEL_K][2] = {0.f};
	if(tile_culling) {
		float3 vert_center_world = {means3D[3 * idx + 0], means3D[3 * idx + 1], means3D[3 * idx + 2]};
		float3 vert_center_view = transformPoint4x3(vert_center_world, viewmatrix);
		for(int ii=0; ii<KERNEL_K; ii++) {
			float u = NEAREST_VERT_RADIUS * local_kernel_vecs[ii].x;
			float v = NEAREST_VERT_RADIUS * local_kernel_vecs[ii].y;
			float3 vert_world = {
				means3D[3 * idx + 0] + u * Rs[0] + v * Rs[1],
				means3D[3 * idx + 1] + u * Rs[3] + v * Rs[4],
				means3D[3 * idx + 2] + u * Rs[6] + v * Rs[7]};

			float3 vert_view = transformPoint4x3(vert_world, viewmatrix);
			adjustVertView(vert_center_view, vert_view);
			float safe_z = copysignf(0.0000001f + fabsf(vert_view.z), vert_view.z);
			float w = vert_view.x / safe_z * focal_x + drkProjectionCenterX((float)W);
			float h = vert_view.y / safe_z * focal_y + drkProjectionCenterY((float)H);
			vert2d[ii][0] = w;
			vert2d[ii][1] = h;
		}
		int visible_pix_num = 0;
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				// Check if the tile is visible
				float minX = (float) (x * BLOCK_X) - .5f - 0.000001f;
				float minY = (float) (y * BLOCK_Y) - .5f - 0.000001f;
				float maxX = (float) (x * BLOCK_X + BLOCK_X - 1) + .5f + 0.000001f;
				float maxY = (float) (y * BLOCK_Y + BLOCK_Y - 1) + .5f + 0.000001f;
				bool visible = doesPolygonIntersectAABB(vert2d, minX, minY, maxX, maxY);
				if(visible)
					visible_pix_num++;
			}
		}
		tiles_touched[idx] = visible_pix_num;
	}
	else
		tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);

	const float op_x = p_orig.x - cam_pos[0];
	const float op_y = p_orig.y - cam_pos[1];
	const float op_z = p_orig.z - cam_pos[2];
	if (store_backward_aux) {
		const int idx_op = idx * 3;
		op[idx_op] = op_x;
		op[idx_op + 1] = op_y;
		op[idx_op + 2] = op_z;
	}
	op_tu[idx] = op_x * Rs[0] + op_y * Rs[3] + op_z * Rs[6];
	op_tv[idx] = op_x * Rs[1] + op_y * Rs[4] + op_z * Rs[7];
	op_n[idx]  = op_x * Rs[2] + op_y * Rs[5] + op_z * Rs[8];
	if (store_backward_aux) {
		const float op_norm = sqrtf(max(op_x * op_x + op_y * op_y + op_z * op_z, 0.0000001f));
		const float op_cos = fabsf(op_n[idx] / op_norm);
		op_cos_n[idx] = op_cos;
#if LOW_PASS_FILTER
		op_inv_cos_n[idx] = __frcp_rn(op_cos);
#endif
	}
#if LOW_PASS_FILTER
	else {
		const float op_norm = sqrtf(max(op_x * op_x + op_y * op_y + op_z * op_z, 0.0000001f));
		const float op_cos = fabsf(op_n[idx] / op_norm);
		op_inv_cos_n[idx] = __frcp_rn(op_cos);
	}
#endif
}


// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching
// and rasterizing data.
template <uint32_t CHANNELS, bool CACHE_SORT, bool AUX_OUTPUT>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ means3D,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* depths,

	const float* __restrict__ rotations,
	const float* __restrict__ scales,
	const float* __restrict__ thetas,
	const float* __restrict__ l1l2_rates,
	const float* __restrict__ opacities,
	const float* __restrict__ acutances,
		const float* __restrict__ cam_pos,
		const float focal_x, float focal_y,
		const float* viewmatrix,

	const float* __restrict__ op_tu,
		const float* __restrict__ op_tv,
	const float* __restrict__ op_n,
	const float* __restrict__ op_cos_n,
	const float* __restrict__ op_inv_cos_n,
	const float2* __restrict__ kernel_vecs,
	const float* __restrict__ scale_inv2,
	const float* __restrict__ kernel_inv_delta,

	float* __restrict__ final_alpha,
	uint32_t* __restrict__ n_contrib,
		const float* __restrict__ bg_color,
		float* __restrict__ out_color,
		float* __restrict__ out_depth,
		float* __restrict__ out_normal)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int    collected_id      [BLOCK_SIZE];
#if LOW_PASS_FILTER || DRK_SCREEN_SPACE_L1_KERNEL
	__shared__ float2 collected_xy      [BLOCK_SIZE];
#endif

	__shared__ float  collected_opacity [BLOCK_SIZE];
#if !DRK_SCREEN_SPACE_L1_KERNEL
	__shared__ float  collected_Rs      [BLOCK_SIZE*9];
#if !DRK_FORCE_ACUTANCE_ONE_KERNEL
	__shared__ float  collected_acutance[BLOCK_SIZE];
#endif
#if !DRK_FORCE_L1_KERNEL
	__shared__ float  collected_l1l2rate[BLOCK_SIZE];
#endif
	__shared__ float  collected_op_tu   [BLOCK_SIZE];
	__shared__ float  collected_op_tv   [BLOCK_SIZE];
	__shared__ float  collected_op_n    [BLOCK_SIZE];
#endif
#if LOW_PASS_FILTER
	__shared__ float  collected_op_inv_cos_n[BLOCK_SIZE];
#endif
#if !DRK_FORCE_L1_KERNEL
	__shared__ float  collected_scale_inv2[BLOCK_SIZE*KERNEL_K];
#endif
#if !DRK_ACU1_L1_ANGLELESS_KERNEL
	__shared__ float  collected_thetas  [BLOCK_SIZE*KERNEL_K];
#endif
#if DRK_SHARE_KERNEL_GEOMETRY
	__shared__ float2 collected_kernel_vecs[BLOCK_SIZE*KERNEL_K];
	__shared__ float  collected_kernel_inv_delta[BLOCK_SIZE*KERNEL_K];
#endif

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	uint32_t n_layers = 0;		// DRK_MAX_LAYERS bounded-OIT layer counter
	float C[CHANNELS] = { 0 };
	float D = 0;
	float N[3] = {0.};

	// CACHE Array to sort the surfels
	float ray_t_array[CACHE_SORT ? CACHE_SIZE : 1] = {0};
	int js[CACHE_SORT ? CACHE_SIZE : 1] = {0};

#if !DRK_SCREEN_SPACE_L1_KERNEL
	// Pre-compute ray direction
	glm::vec3 dir_camera = {(pixf.x - drkRayCenterX((float)W)) / focal_x, (pixf.y - drkRayCenterY((float)H)) / focal_y, 1.0f};
	float dir_norm = sqrt(dir_camera[0] * dir_camera[0] + dir_camera[1] * dir_camera[1] + dir_camera[2] * dir_camera[2]);
	for (int dir_idx=0; dir_idx < 3; dir_idx++)
		dir_camera[dir_idx] = dir_camera[dir_idx] / dir_norm;
	glm::vec3 dir = {
		viewmatrix[0] * dir_camera[0] + viewmatrix[1] * dir_camera[1] + viewmatrix[2] * dir_camera[2],
		viewmatrix[4] * dir_camera[0] + viewmatrix[5] * dir_camera[1] + viewmatrix[6] * dir_camera[2],
		viewmatrix[8] * dir_camera[0] + viewmatrix[9] * dir_camera[1] + viewmatrix[10] * dir_camera[2]
	};
	float zscale = fabsf(dir_camera[2]);
#endif

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
			if (range.x + progress < range.y)
			{
				int coll_id = point_list[range.x + progress];
				if(coll_id >= 0 && opacities[coll_id] < 1.0f / 255.0f)
					coll_id = -1;
				collected_id[block.thread_rank()] = coll_id;

				if(coll_id >= 0) {
#if LOW_PASS_FILTER || DRK_SCREEN_SPACE_L1_KERNEL
				collected_xy[block.thread_rank()] = points_xy_image[coll_id];
#endif
				collected_opacity[block.thread_rank()] = opacities[coll_id];
#if !DRK_SCREEN_SPACE_L1_KERNEL
				for (int i = 0; i < 9; i++)
					collected_Rs[i + 9 * block.thread_rank()] = rotations[coll_id * 9 + i];
#if !DRK_FORCE_ACUTANCE_ONE_KERNEL
				collected_acutance[block.thread_rank()] = acutances[coll_id];
#endif
#if !DRK_FORCE_L1_KERNEL
				collected_l1l2rate[block.thread_rank()] = l1l2_rates[coll_id];
#endif
				collected_op_tu[block.thread_rank()] = op_tu[coll_id];
				collected_op_tv[block.thread_rank()] = op_tv[coll_id];
				collected_op_n[block.thread_rank()] = op_n[coll_id];
#endif
#if LOW_PASS_FILTER
				collected_op_inv_cos_n[block.thread_rank()] = op_inv_cos_n[coll_id];
#endif
#if !DRK_FORCE_L1_KERNEL
				for (int i = 0; i < KERNEL_K; i++)
					collected_scale_inv2[i + KERNEL_K * block.thread_rank()] = scale_inv2[coll_id * KERNEL_K + i];
#endif
#if !DRK_ACU1_L1_ANGLELESS_KERNEL
				for (int i = 0; i < KERNEL_K; i++)
					collected_thetas[i + KERNEL_K * block.thread_rank()] = thetas[coll_id * KERNEL_K + i];
#endif
#if DRK_SHARE_KERNEL_GEOMETRY
				for (int i = 0; i < KERNEL_K; i++)
					collected_kernel_vecs[i + KERNEL_K * block.thread_rank()] = kernel_vecs[coll_id * KERNEL_K + i];
				for (int i = 0; i < KERNEL_K; i++)
					collected_kernel_inv_delta[i + KERNEL_K * block.thread_rank()] = kernel_inv_delta[coll_id * KERNEL_K + i];
#endif
			}
		}
		block.sync();

		// Initialize CACHE Array to sort the surfels
		int next_array[CACHE_SORT ? CACHE_SIZE : 1] = {0};
		int prev_array[CACHE_SORT ? CACHE_SIZE : 1] = {0};
		if(CACHE_SORT) {
			for(int ii=0; ii<CACHE_SIZE; ii++) {
				js[ii] = -1;
				next_array[ii] = -1;
				prev_array[ii] = -1;
			}
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
#if DRK_SCREEN_SPACE_L1_KERNEL
			j = jj;
#else
			if(CACHE_SORT) {
				// There are unprocessed surfels
				if(jj < min(BLOCK_SIZE, toDo)) {
					// Pre-calculate Intersection (dir, cam_pos -> uv)
					const float * Rs = collected_Rs + jj * 9;
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
#endif

			// process in cache sorted order
			if (j < 0)
				continue;

			float opacity = collected_opacity[j];
			if (opacity < 1.0f / 255.0f)
				continue;

			float alpha = 0.f;
#if DRK_SCREEN_SPACE_L1_KERNEL
#if DRK_SHARE_KERNEL_GEOMETRY
			const float2 * polygon_xy = collected_kernel_vecs + KERNEL_K*j;
#else
			const float2 * polygon_xy = kernel_vecs + KERNEL_K*collected_id[j];
#endif
			float2 center_xy = collected_xy[j];
			bool inside_polygon = false;
#pragma unroll
			for (int idx = 0; idx < KERNEL_K; ++idx) {
				float2 a = center_xy;
				float2 b = polygon_xy[idx];
				float2 c = polygon_xy[idx == (KERNEL_K - 1) ? 0 : idx + 1];
				float cross_ab = (b.x - a.x) * (pixf.y - a.y) - (b.y - a.y) * (pixf.x - a.x);
				float cross_bc = (c.x - b.x) * (pixf.y - b.y) - (c.y - b.y) * (pixf.x - b.x);
				float cross_ca = (a.x - c.x) * (pixf.y - c.y) - (a.y - c.y) * (pixf.x - c.x);
				bool has_pos = (cross_ab > 0.0001f) || (cross_bc > 0.0001f) || (cross_ca > 0.0001f);
				bool has_neg = (cross_ab < -0.0001f) || (cross_bc < -0.0001f) || (cross_ca < -0.0001f);
				inside_polygon |= !(has_pos && has_neg);
			}
			if (!inside_polygon)
				continue;
			alpha = opacity;
#else
			// Step 1. Intersection (dir, cam_pos -> uv)
			const float * Rs = collected_Rs + j * 9;
			const glm::vec3 normal = {Rs[2], Rs[5], Rs[8]};
			float dir_dot_n = Rs[2] * dir[0] + Rs[5] * dir[1] + Rs[8] * dir[2];
			float ray_t = collected_op_n[j] / dir_dot_n;
			if (fabsf(dir_dot_n) < 0.0001f)
				continue;
			if (ray_t < 0.001f)
				continue;
			float dir_dot_tu = dir[0] * Rs[0] + dir[1] * Rs[3] + dir[2] * Rs[6];
			float dir_dot_tv = dir[0] * Rs[1] + dir[1] * Rs[4] + dir[2] * Rs[7];
			float2 uv = {ray_t*dir_dot_tu-collected_op_tu[j], ray_t*dir_dot_tv-collected_op_tv[j]};

			// Step 2. Determine the interval
#if DRK_SHARE_KERNEL_GEOMETRY
			const float2 * kernel_vec = collected_kernel_vecs + KERNEL_K*j;
#else
			const float2 * kernel_vec = kernel_vecs + KERNEL_K*collected_id[j];
#endif
#if DRK_ACU1_L1_ANGLELESS_KERNEL
			int k = 0;
#pragma unroll
			for (int idx = 1; idx < KERNEL_K; ++idx)
				k += drk_angle_less_0_2pi(kernel_vec[idx], uv);
#else
			float theta = atan2f(uv.y, uv.x);
			theta = theta < 0.f? theta + 2.0f * PI: theta;
			theta = fminf(theta * (0.5f / PI), 1.0f);
			const float * thetas_array = collected_thetas + KERNEL_K*j;
			int k = branchless_segment_search(theta, thetas_array);
#endif
#if !DRK_FORCE_L1_KERNEL
			const float * scale_inv2_ptr = collected_scale_inv2 + KERNEL_K*j;
			float uv_l2norm = uv.x * uv.x + uv.y * uv.y;
			// Numerical Protection!
			uv_l2norm = max(uv_l2norm, 0.00000001f);
			float theta_l = k==0? 0.0f: thetas_array[k-1];
			float theta_r = thetas_array[k];
			float inv_theta_span = __frcp_rn(theta_r - theta_l);
			float linear_rate = (theta - theta_l) * inv_theta_span;
			float rate = 0.5f * (__cosf((1.0f - linear_rate) * PI) + 1.0f);
			float inv_sl2 = scale_inv2_ptr[k];
			float inv_sr2 = k==(KERNEL_K-1)? scale_inv2_ptr[0]: scale_inv2_ptr[k+1];
#endif

			// Step 3. Affine transformation
#if DRK_SHARE_KERNEL_GEOMETRY
			float inv_delta = collected_kernel_inv_delta[KERNEL_K*j + k];
#else
			float inv_delta = kernel_inv_delta[KERNEL_K*collected_id[j] + k];
#endif
			float2 e1 = kernel_vec[k];
			float2 e2 = kernel_vec[k==(KERNEL_K-1)? 0: k+1];
			float2 uv_t = {(e2.y * uv.x - e2.x * uv.y) * inv_delta, (- e1.y * uv.x + e1.x * uv.y) * inv_delta};

			// Step 4. Kernel function (theta, uv_norm -> kernel_opacity)
			float uvt_l1norm = fabsf(uv_t.x) + fabsf(uv_t.y);
			uvt_l1norm = uvt_l1norm * uvt_l1norm * 0.5f;
			uvt_l1norm = fmaxf(uvt_l1norm, 0.00000001f);
#if DRK_FORCE_L1_KERNEL
			float uv_norm = uvt_l1norm;
#else
			float l1l2_rate = collected_l1l2rate[j];
			float uvt_l2norm = uv_l2norm * (rate * inv_sr2 + (1.0f - rate) * inv_sl2) * 0.5f;
			uvt_l2norm = fmaxf(uvt_l2norm, 0.00000001f);
			float uv_norm = l1l2_rate * uvt_l1norm + (1.0f - l1l2_rate) * uvt_l2norm;
#endif
			// Step 5. Sharpening
#if DRK_FORCE_ACUTANCE_ONE_KERNEL
			if (uv_norm > DRK_LN2)
				continue;
			alpha = opacity;
#else
			float kernel_opacity = __expf(- uv_norm);
			const float k_acu_raw = collected_acutance[j];
			float sharpen = 0.f;
			if(SHARPEN_ALPHA){
				const float k_acu = fminf(k_acu_raw, 0.999999f);
				alpha = kernel_opacity * opacity;
				sharpen = (alpha < ((1.0f + k_acu) / 4.0f))? ((1.0f - k_acu) / (1.0f + k_acu) * alpha): ((1.0f + k_acu) / (1.0f - k_acu) * alpha - k_acu / (1.0f - k_acu));
				sharpen = (alpha < ((3.0f - k_acu) / 4.0f))? sharpen: ((1.0f - k_acu) / (1.0f + k_acu) * alpha + 2.0f * k_acu / (1.0f + k_acu));
				alpha = sharpen;
			}
			else if (k_acu_raw >= 1.0f - DRK_ACUTANCE_ONE_EPS) {
				sharpen = kernel_opacity >= 0.5f ? 1.0f : 0.0f;
				alpha = sharpen * opacity;
			}
			else{
				const float k_acu = fminf(k_acu_raw, 0.999999f);
				sharpen = (kernel_opacity < ((1.0f + k_acu) / 4.0f))? ((1.0f - k_acu) / (1.0f + k_acu) * kernel_opacity): ((1.0f + k_acu) / (1.0f - k_acu) * kernel_opacity - k_acu / (1.0f - k_acu));
				sharpen = (kernel_opacity < ((3.0f - k_acu) / 4.0f))? sharpen: ((1.0f - k_acu) / (1.0f + k_acu) * kernel_opacity + 2.0f * k_acu / (1.0f + k_acu));
				alpha = sharpen * opacity;
			}
#endif

			// Step 6. Low pass filter
			bool filter_cond = false;
#if LOW_PASS_FILTER
			float2 xy = collected_xy[j];
			float inv_cos_dn = collected_op_inv_cos_n[j];
			float2 res_2d = {(xy.x - pixf.x) * inv_cos_dn, (xy.y - pixf.y) * inv_cos_dn};
			float dist_2d_norm = FilterInvSquare * (res_2d.x * res_2d.x + res_2d.y * res_2d.y);
			float G_lps = __expf(-0.5f * dist_2d_norm);
			float alpha_lpf = opacity * G_lps;
			filter_cond = alpha < alpha_lpf;
			alpha = filter_cond? alpha_lpf: alpha;
#endif
#endif
			if (alpha < 1.0f / 255.0f)
				continue;

			// Alpha blending
			alpha = min(0.99f, alpha);
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)  // Only when T < 0.01, since alpha <= 0.99
			{
				done = true;
				continue;
			}
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
#if !DRK_SCREEN_SPACE_L1_KERNEL
			if constexpr (AUX_OUTPUT) {
				float pix_depth = filter_cond? depths[collected_id[j]]: ray_t * zscale;
				D += pix_depth * alpha * T;
				for(int ch=0; ch < 3; ch++)
					N[ch] += normal[ch] * copysignf(1., -dir_dot_n) * alpha * T;
			}
#endif
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
#if DRK_MAX_LAYERS > 0
			// Bounded OIT: stop after N contributing (nearest, front-to-back) layers
			if (++n_layers >= DRK_MAX_LAYERS)
				done = true;
#endif
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
#if !DRK_SCREEN_SPACE_L1_KERNEL
		if constexpr (AUX_OUTPUT) {
			final_alpha[pix_id] = max(0.f, 1.f - T);
			n_contrib[pix_id] = last_contributor;
			out_depth[pix_id] = D;
			for (int ch = 0; ch < 3; ch ++)
				out_normal[ch * H * W + pix_id] = N[ch];
		}
#endif
	}
}

void FORWARD::render(
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

	float * op_tu,
	float * op_tv,
	float * op_n,
	const float * op_cos_n,
	const float * op_inv_cos_n,
	const float2 * kernel_vecs,
	const float * scale_inv2,
	const float * kernel_inv_delta,

	float* final_alpha,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_depth,
	float* out_normal,

	bool cache_sort,
	bool render_aux)
{
#if DRK_SCREEN_SPACE_L1_KERNEL
	renderCUDA<NUM_CHANNELS, false, false> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means3D,
		means2D,
		colors,
		depths,

		rotations,
		scales,
		thetas,
		l1l2_rates,
		opacities,
		acutances,
		cam_pos,
		focal_x, focal_y,
		viewmatrix,

		op_tu,
		op_tv,
		op_n,
		op_cos_n,
		op_inv_cos_n,
		kernel_vecs,
		scale_inv2,
		kernel_inv_delta,

		final_alpha,
		n_contrib,
		bg_color,
		out_color,
		out_depth,
		out_normal);
#else
	if (cache_sort && render_aux)
		renderCUDA<NUM_CHANNELS, true, true> << <grid, block >> > (
			ranges,
			point_list,
		W, H,
		means3D,
		means2D,
		colors,
		depths,

		rotations,
		scales,
		thetas,
		l1l2_rates,
		opacities,
		acutances,
		cam_pos,
		focal_x, focal_y,
		viewmatrix,

		op_tu,
		op_tv,
		op_n,
		op_cos_n,
		op_inv_cos_n,
		kernel_vecs,
		scale_inv2,
		kernel_inv_delta,

		final_alpha,
		n_contrib,
			bg_color,
			out_color,
			out_depth,
			out_normal);
	else if (cache_sort)
		renderCUDA<NUM_CHANNELS, true, false> << <grid, block >> > (
			ranges,
			point_list,
			W, H,
			means3D,
			means2D,
			colors,
			depths,

			rotations,
			scales,
			thetas,
			l1l2_rates,
			opacities,
			acutances,
			cam_pos,
			focal_x, focal_y,
			viewmatrix,

			op_tu,
			op_tv,
			op_n,
			op_cos_n,
			op_inv_cos_n,
			kernel_vecs,
			scale_inv2,
			kernel_inv_delta,

			final_alpha,
			n_contrib,
			bg_color,
			out_color,
			out_depth,
			out_normal);
	else if (render_aux)
		renderCUDA<NUM_CHANNELS, false, true> << <grid, block >> > (
			ranges,
			point_list,
			W, H,
			means3D,
			means2D,
			colors,
			depths,

			rotations,
			scales,
			thetas,
			l1l2_rates,
			opacities,
			acutances,
			cam_pos,
			focal_x, focal_y,
			viewmatrix,

			op_tu,
			op_tv,
			op_n,
			op_cos_n,
			op_inv_cos_n,
			kernel_vecs,
			scale_inv2,
			kernel_inv_delta,

			final_alpha,
			n_contrib,
			bg_color,
			out_color,
			out_depth,
			out_normal);
	else
		renderCUDA<NUM_CHANNELS, false, false> << <grid, block >> > (
			ranges,
			point_list,
			W, H,
			means3D,
			means2D,
			colors,
			depths,

			rotations,
			scales,
			thetas,
			l1l2_rates,
			opacities,
			acutances,
			cam_pos,
			focal_x, focal_y,
			viewmatrix,

			op_tu,
			op_tv,
			op_n,
			op_cos_n,
			op_inv_cos_n,
			kernel_vecs,
			scale_inv2,
			kernel_inv_delta,

			final_alpha,
			n_contrib,
			bg_color,
			out_color,
			out_depth,
			out_normal);
#endif
}


void FORWARD::preprocess2D(int P, int D, int M,
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
	float * op_cos_n,
	float * op_inv_cos_n,
	float2 * kernel_vecs,
	float * scale_inv2,
	float * kernel_inv_delta,
	uint2* rect_min,
	uint2* rect_max,

	bool prefiltered,
	bool tile_culling,
	bool store_backward_aux)
{
	preprocess2DCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		thetas,
		opacities,
		acutances,
		scale_modifier,
		rotations,
		shs,
		clamped,
		colors_precomp,
		viewmatrix,
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		rgb,
		grid,
		tiles_touched,

		op,
		op_tu,
		op_tv,
		op_n,
		op_cos_n,
		op_inv_cos_n,
		kernel_vecs,
		scale_inv2,
		kernel_inv_delta,
		rect_min,
		rect_max,

		prefiltered,
		tile_culling,
		store_backward_aux
			);
}
