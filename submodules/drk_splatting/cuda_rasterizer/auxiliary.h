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

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)


__device__ const float FilterInvSquare = 16.0;  // (1 / s^2)


// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ void swap(float &a, float &b) {
    float temp = a;
    a = b;
    b = temp;
}

__forceinline__ __device__ void swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.01f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

__forceinline__ __device__ int kernel_search_sorted(const float x, const float* y, int range=KERNEL_K) {
	if (y[0] >= x)
		return 0;
	else if (y[range-1] <= x)
		return range - 1;
	// make sure y[low] < x < y[high]
	int low = 0, high = range - 1;
    while (low < high-1) {
        int mid = low + (high - low) / 2;
        if (y[mid] == x) {
            return mid;
        } else if (y[mid] < x) {
            low = mid;
        } else {
            high = mid;
        }
    }
    return low + 1;
}

__forceinline__ __device__ float3 calculate_rayt(const float* viewmatrix, const float* cam_pos, const float* mean3D, const float focal_x, const float focal_y, const float* Rs, float2 pixf, float W, float H, bool filter_side, glm::vec3& intersect) {
	glm::vec3 dir_camera = {(pixf.x - (W-1.0) * 0.5) / focal_x, (pixf.y - (H-1.0) * 0.5) / focal_y, 1.0f};
	float dir_norm = sqrt(dir_camera[0] * dir_camera[0] + dir_camera[1] * dir_camera[1] + dir_camera[2] * dir_camera[2]);
	for (int dir_idx=0; dir_idx < 3; dir_idx++)
		dir_camera[dir_idx] = dir_camera[dir_idx] / dir_norm;
	glm::vec3 dir = {
		viewmatrix[0] * dir_camera[0] + viewmatrix[1] * dir_camera[1] + viewmatrix[2] * dir_camera[2],
		viewmatrix[4] * dir_camera[0] + viewmatrix[5] * dir_camera[1] + viewmatrix[6] * dir_camera[2],
		viewmatrix[8] * dir_camera[0] + viewmatrix[9] * dir_camera[1] + viewmatrix[10] * dir_camera[2]
	};
	const glm::vec3 normal = {Rs[2], Rs[5], Rs[8]};
	float cos_ndir = normal[0] * dir[0] + normal[1] * dir[1] + normal[2] * dir[2];
	cos_ndir = fabsf(cos_ndir);
	if (filter_side && cos_ndir < 0.000001f) {
		float3 rayt_zscale_cosndir = {10000000.f, fabsf(dir_camera[2]), cos_ndir};
		return rayt_zscale_cosndir;
	}
	else {
		const glm::vec3 gs_dir = {mean3D[0] - cam_pos[0], mean3D[1] - cam_pos[1], mean3D[2] - cam_pos[2]};
		float ray_t = (normal[0] * gs_dir[0] + normal[1] * gs_dir[1] + normal[2] * gs_dir[2]) / (normal[0] * dir[0] + normal[1] * dir[1] + normal[2] * dir[2]);
		for (int i=0; i<3; i++)
			intersect[i] = cam_pos[i] + ray_t * dir[i];
		float3 rayt_zscale_cosndir = {ray_t, fabsf(dir_camera[2]), cos_ndir};
		return rayt_zscale_cosndir;
	}
}

__forceinline__ __device__ float3 calculate_rayt(const float* viewmatrix, const float* cam_pos, const float* mean3D, const float focal_x, const float focal_y, const float* Rs, float2 pixf, float W, float H, bool filter_side) {
	glm::vec3 dir_camera = {(pixf.x - (W-1.0) * 0.5) / focal_x, (pixf.y - (H-1.0) * 0.5) / focal_y, 1.0f};
	float dir_norm = sqrt(dir_camera[0] * dir_camera[0] + dir_camera[1] * dir_camera[1] + dir_camera[2] * dir_camera[2]);
	for (int dir_idx=0; dir_idx < 3; dir_idx++)
		dir_camera[dir_idx] = dir_camera[dir_idx] / dir_norm;
	glm::vec3 dir = {
		viewmatrix[0] * dir_camera[0] + viewmatrix[1] * dir_camera[1] + viewmatrix[2] * dir_camera[2],
		viewmatrix[4] * dir_camera[0] + viewmatrix[5] * dir_camera[1] + viewmatrix[6] * dir_camera[2],
		viewmatrix[8] * dir_camera[0] + viewmatrix[9] * dir_camera[1] + viewmatrix[10] * dir_camera[2]
	};
	const glm::vec3 normal = {Rs[2], Rs[5], Rs[8]};
	float cos_ndir = normal[0] * dir[0] + normal[1] * dir[1] + normal[2] * dir[2];
	cos_ndir = fabsf(cos_ndir);
	if (filter_side && cos_ndir < 0.000001f) {
		float3 rayt_zscale_cosndir = {10000000.f, fabsf(dir_camera[2]), cos_ndir};
		return rayt_zscale_cosndir;
	}
	else {
		const glm::vec3 gs_dir = {mean3D[0] - cam_pos[0], mean3D[1] - cam_pos[1], mean3D[2] - cam_pos[2]};
		float ray_t = (normal[0] * gs_dir[0] + normal[1] * gs_dir[1] + normal[2] * gs_dir[2]) / (normal[0] * dir[0] + normal[1] * dir[1] + normal[2] * dir[2]);
		float3 rayt_zscale_cosndir = {ray_t, dir_camera[2], cos_ndir};
		return rayt_zscale_cosndir;
	}
}

__forceinline__ __device__ int push_and_pop_array(int* gs_idx_array, float* depth_array, int* next_array, int* prev_array, int& head, int& tail, int& length, int& last_cursor, const int new_gs_idx, const float new_depth) {
	// If the array is empty
	if(length == 0) {
		// If the new point is invalid, return -2
		if(new_gs_idx==-1)
			return -2;
		// If the new point is valid, insert it to 0
		last_cursor = 0;
		gs_idx_array[0] = new_gs_idx;
		depth_array[0] = new_depth;
		next_array[0] = -1;
		prev_array[0] = -1;
		head = 0;
		tail = 0;
		length = 1;
		return -1;
	}

	// If the new point is invalid and the chain is not empty, pop the head
	if(new_gs_idx==-1){
		int gs_idx = gs_idx_array[head];
		length --;
		int old_head = head;
		head = next_array[head];
		next_array[old_head]   = -1;
		prev_array[old_head]   = -1;
		gs_idx_array[old_head] = -1;
		prev_array[head] = -1;
		last_cursor = head;
		return gs_idx;
	}

	// Find the place to store the new point and the index to return
	int store_idx = 0;
	int return_idx;
	int new_head;
	bool pop_head;
	if(length==CACHE_SIZE) {
		store_idx = head;
		return_idx = gs_idx_array[head];
		new_head = next_array[head];
		pop_head = true;
	}
	else {
		while(gs_idx_array[store_idx]!=-1)
			store_idx ++;
		return_idx = -1;
		new_head = head;
		length ++;
		pop_head = false;
	}
	
	// Initialize the scan cursor
	int scan_cursor = last_cursor;
	bool scan_to_tail;
	int scan_cursor_end;
	if(new_depth > depth_array[scan_cursor]) {
		scan_to_tail = true;
		scan_cursor_end = tail;
	}
	else {
		scan_to_tail = false;
		scan_cursor_end = head;
	}
	last_cursor = store_idx;

	// Scan the array to find the position to insert the new point
	while((new_depth <= depth_array[scan_cursor]) != scan_to_tail) {
		if(scan_cursor==scan_cursor_end)
			break;
		scan_cursor = scan_to_tail? next_array[scan_cursor] : prev_array[scan_cursor];
	}

	// If scan to the end of the chain, insert the point to the end (head/tail)
	if((new_depth <= depth_array[scan_cursor]) != scan_to_tail){
		if(scan_to_tail) {
			next_array[tail] = store_idx;
			prev_array[store_idx] = tail;
			next_array[store_idx] = -1;
			tail = store_idx;
			head = new_head;
		}
		else {
			if(pop_head)
				return new_gs_idx;
			prev_array[head] = store_idx;
			next_array[store_idx] = (store_idx==head)? next_array[head]: head;
			prev_array[store_idx] = -1;
			head = store_idx;
		}
	}
	// If the new point is to be inserted in the middle of the chain
	else {
		// Insert the new point to the left of the scan_cursor
		if(scan_to_tail) {
			next_array[store_idx] = scan_cursor;
			prev_array[store_idx] = prev_array[scan_cursor];
			if(prev_array[scan_cursor]!=store_idx)
				next_array[prev_array[scan_cursor]] = store_idx;
			prev_array[scan_cursor] = store_idx;
			head = (scan_cursor==new_head)? store_idx: new_head;
		}
		// Insert the new point to the right of the scan_cursor
		else {
			prev_array[store_idx] = scan_cursor;
			next_array[store_idx] = next_array[scan_cursor];
			prev_array[next_array[scan_cursor]] = store_idx;
			if(scan_cursor!=store_idx)
				next_array[scan_cursor] = store_idx;
			head = (scan_cursor==head && pop_head)? store_idx: new_head;
		}
	}
	// Set the gs_idx and depth values of the new point
	gs_idx_array[store_idx] = new_gs_idx;
	depth_array[store_idx] = new_depth;

	// Return the index to pop
	return return_idx;
}

__forceinline__ __device__  bool segmentIntersectsAABB(
    float rayStartX, float rayStartY,
    float rayEndX, float rayEndY,
    float boxMinX, float boxMinY,
    float boxMaxX, float boxMaxY,
    float &intersectionX, float &intersectionY) {
    
    float dirX = rayEndX - rayStartX;
    float dirY = rayEndY - rayStartY;

    float tmin = (boxMinX - rayStartX) / dirX;
    float tmax = (boxMaxX - rayStartX) / dirX;
    if (tmin > tmax) swap(tmin, tmax);

    float tymin = (boxMinY - rayStartY) / dirY;
    float tymax = (boxMaxY - rayStartY) / dirY;
    if (tymin > tymax) swap(tymin, tymax);

    if ((tmin > tymax) || (tymin > tmax))
        return false;

    if (tymin > tmin)
        tmin = tymin;

    if (tmin < 0 || tmin > 1)
        return false;

    float t = tmin;
    intersectionX = rayStartX + t * dirX;
    intersectionY = rayStartY + t * dirY;

    return true;
}

// Function to check if a point is inside the AABB
__forceinline__ __device__ bool isPointInsideAABB(float px, float py, float minX, float minY, float maxX, float maxY) {
    return (px >= minX && px <= maxX && py >= minY && py <= maxY);
}

// Function to check if two line segments intersect
__forceinline__ __device__ bool doLinesIntersect(float p1x, float p1y, float q1x, float q1y, float p2x, float p2y, float q2x, float q2y) {
    auto orientation = [](float ax, float ay, float bx, float by, float cx, float cy) {
        float val = (by - ay) * (cx - bx) - (bx - ax) * (cy - by);
        if (val == 0) return 0;  // collinear
        return (val > 0) ? 1 : 2; // clock or counterclock wise
    };

    int o1 = orientation(p1x, p1y, q1x, q1y, p2x, p2y);
    int o2 = orientation(p1x, p1y, q1x, q1y, q2x, q2y);
    int o3 = orientation(p2x, p2y, q2x, q2y, p1x, p1y);
    int o4 = orientation(p2x, p2y, q2x, q2y, q1x, q1y);

    if (o1 != o2 && o3 != o4) return true;

    return false;
}

// Function to check if a point is inside a polygon using the ray-casting method
__forceinline__ __device__ bool isPointInsidePolygon(float px, float py, float polygon[][2]) {
    int count = 0;
    for (int i = 0; i < KERNEL_K; ++i) {
        float p1x = polygon[i][0], p1y = polygon[i][1];
        float p2x = polygon[(i + 1) % KERNEL_K][0], p2y = polygon[(i + 1) % KERNEL_K][1];

        if ((p1y > py) != (p2y > py) &&
            px < (p2x - p1x) * (py - p1y) / (p2y - p1y) + p1x) {
            count = !count;
        }
    }
    return count;
}

// Function to check if the polygon intersects the AABB
__forceinline__ __device__ bool doesPolygonIntersectAABB(float polygon[][2], float minX, float minY, float maxX, float maxY) {
    // Check if any polygon vertex is inside the AABB
    for (int i = 0; i < KERNEL_K; ++i) {
        if (isPointInsideAABB(polygon[i][0], polygon[i][1], minX, minY, maxX, maxY)) {
            return true;
        }
    }

    // Define AABB corner points
    float boxCorners[4][2] = {
        {minX, minY},
        {maxX, minY},
        {maxX, maxY},
        {minX, maxY}
    };

    // Check if any AABB corner is inside the polygon
    for (int i = 0; i < 4; ++i) {
        if (isPointInsidePolygon(boxCorners[i][0], boxCorners[i][1], polygon)) {
            return true;
        }
    }

    // Define AABB edges
    float boxEdges[4][4] = {
        {minX, minY, maxX, minY},
        {maxX, minY, maxX, maxY},
        {maxX, maxY, minX, maxY},
        {minX, maxY, minX, minY}
    };

    // Check for intersection between polygon edges and AABB edges
    for (int i = 0; i < KERNEL_K; ++i) {
        float p1x = polygon[i][0], p1y = polygon[i][1];
        float p2x = polygon[(i + 1) % KERNEL_K][0], p2y = polygon[(i + 1) % KERNEL_K][1];

        for (int j = 0; j < 4; ++j) {
            if (doLinesIntersect(p1x, p1y, p2x, p2y, boxEdges[j][0], boxEdges[j][1], boxEdges[j][2], boxEdges[j][3])) {
                return true;
            }
        }
    }

    return false;
}


__forceinline__ __device__ void adjustVertView(float3& vert_center_view, float3& vert_view) {
    if (vert_view.z < 0.0000001f) {
        // Calculate the line direction
        float3 direction = {
            vert_view.x - vert_center_view.x,
            vert_view.y - vert_center_view.y,
            vert_view.z - vert_center_view.z
        };

        // Calculate the scale factor to make vert_view.z = 0.001
        float scale = (0.001f - vert_center_view.z) / direction.z;

        // Update vert_view position
        vert_view.x = vert_center_view.x + scale * direction.x;
        vert_view.y = vert_center_view.y + scale * direction.y;
        vert_view.z = 0.001f;
    }
}


#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif