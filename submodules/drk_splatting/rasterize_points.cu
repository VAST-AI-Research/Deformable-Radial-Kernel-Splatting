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
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& thetas,
	const torch::Tensor& l1l2_rates,
	const torch::Tensor& rotations,
	const torch::Tensor& acutances,
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool cache_sort,
	const bool tile_culling,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_normal = torch::full({3, H, W}, 0.0, float_opts);
  torch::Tensor out_alpha = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		thetas.contiguous().data_ptr<float>(),
		l1l2_rates.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		acutances.contiguous().data_ptr<float>(),
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_depth.contiguous().data<float>(),
		out_normal.contiguous().data<float>(),
		out_alpha.contiguous().data<float>(),

		cache_sort,
		tile_culling,
		radii.contiguous().data<int>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, out_depth, out_normal, out_alpha, radii, geomBuffer, binningBuffer, imgBuffer);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,

	const torch::Tensor& scales,
	const torch::Tensor& thetas,
	const torch::Tensor& l1l2_rates,

	const torch::Tensor& rotations,
	const torch::Tensor& acutances,
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,

    const torch::Tensor& out_color,
    const torch::Tensor& out_alpha,
    const torch::Tensor& out_depth,
    const torch::Tensor& out_normal,

    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dout_alpha,
    const torch::Tensor& dL_ddepths,
    const torch::Tensor& dL_dnormal,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,

	const bool cache_sort,
	const bool tile_culling,
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D_densify = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dopacity_densify = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  
  torch::Tensor dL_dscales = torch::zeros({P, KERNEL_K}, means3D.options());
  torch::Tensor dL_dthetas = torch::zeros({P, KERNEL_K}, means3D.options());
  torch::Tensor dL_dl1l2_rates = torch::zeros({P, 1}, means3D.options());

  torch::Tensor dL_drotations = torch::zeros({P, 9}, means3D.options());
  torch::Tensor dL_dacutances= torch::zeros({P, 1}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  opacity.contiguous().data<float>(), 

	  scales.contiguous().data_ptr<float>(),
	  thetas.contiguous().data_ptr<float>(),
	  l1l2_rates.contiguous().data_ptr<float>(),

	  scale_modifier,
	  rotations.data_ptr<float>(),
	  acutances.data_ptr<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),

	  out_color.contiguous().data<float>(),
	  out_alpha.contiguous().data<float>(),
	  out_depth.contiguous().data<float>(),
	  out_normal.contiguous().data<float>(),

	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_alpha.contiguous().data<float>(),
	  dL_ddepths.contiguous().data<float>(),
	  dL_dnormal.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dmeans2D_densify.contiguous().data<float>(),
	  dL_dopacity_densify.contiguous().data<float>(),
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),

	  dL_dscales.contiguous().data<float>(),
	  dL_dthetas.contiguous().data<float>(),
	  dL_dl1l2_rates.contiguous().data<float>(),

	  dL_drotations.contiguous().data<float>(),
	  dL_dacutances.contiguous().data<float>(),

	  cache_sort,
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dmeans2D_densify, dL_dopacity_densify, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dsh, dL_dscales, dL_dthetas, dL_dl1l2_rates, dL_drotations, dL_dacutances);
}

