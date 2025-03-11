#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class DoGFilter(nn.Module):
    def __init__(self, channels, sigma1):
        super(DoGFilter, self).__init__()
        self.channels = channels
        self.sigma1 = sigma1
        self.sigma2 = 2 * sigma1  # Ensure the 1:2 ratio
        self.kernel_size1 = int(2 * round(3 * self.sigma1) + 1)
        self.kernel_size2 = int(2 * round(3 * self.sigma2) + 1)
        self.padding1 = (self.kernel_size1 - 1) // 2
        self.padding2 = (self.kernel_size2 - 1) // 2
        self.weight1 = self.get_gaussian_kernel(self.kernel_size1, self.sigma1)
        self.weight2 = self.get_gaussian_kernel(self.kernel_size2, self.sigma2)


    def get_gaussian_kernel(self, kernel_size, sigma):
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma**2.
        
        kernel = torch.exp(-(xy_grid - mean).pow(2).sum(dim=-1) / (2 * variance))
        kernel = kernel / kernel.sum()  # Normalize the kernel
        kernel = kernel.repeat(self.channels, 1, 1, 1)
        
        return kernel

    @torch.no_grad()
    def forward(self, x):
        gaussian1 = F.conv2d(x, self.weight1.to(x.device), bias=None, stride=1, padding=self.padding1, groups=self.channels)
        gaussian2 = F.conv2d(x, self.weight2.to(x.device), bias=None, stride=1, padding=self.padding2, groups=self.channels)
        return gaussian1 - gaussian2
    
    
def apply_dog_filter(batch, freq=50, scale_factor=0.5):
    """
    Apply a Difference of Gaussian filter to a batch of images.
    
    Args:
        batch: torch.Tensor, shape (B, C, H, W)
        freq: Control variable ranging from 0 to 100.
              - 0 means original image
              - 1.0 means smoother difference
              - 100 means sharpest difference
        scale_factor: Factor by which the image is downscaled before applying DoG.
    
    Returns:
        torch.Tensor: Processed image using DoG.
    """
    # Convert to grayscale if it's a color image
    if batch.size(1) == 3:
        batch = torch.mean(batch, dim=1, keepdim=True)

    # Downscale the image
    downscaled = F.interpolate(batch, scale_factor=scale_factor, mode='bilinear', align_corners=False)

    channels = downscaled.size(1)

    # Set sigma1 value based on freq parameter. sigma2 will be 2*sigma1.
    sigma1 = 0.1 + (100 - freq) * 0.1 if freq >=50 else 0.1 + freq * 0.1

    dog_filter = DoGFilter(channels, sigma1)
    mask = dog_filter(downscaled)

    # Upscale the mask back to original size
    upscaled_mask = F.interpolate(mask, size=batch.shape[-2:], mode='bilinear', align_corners=False)

    upscaled_mask = upscaled_mask - upscaled_mask.min()
    upscaled_mask = upscaled_mask / upscaled_mask.max() if freq >=50 else  1.0 - upscaled_mask / upscaled_mask.max()
    
    upscaled_mask = (upscaled_mask >=0.5).to(torch.float)
    return upscaled_mask[:,0,...]

def apply_sobel_filter(image_tensor):
    """ Apply Sobel filter to detect edges in the image. """
    # Sobel kernels for x and y directions
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3).expand([3, 3, 3, 3]).to(image_tensor.device)
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3).expand([3, 3, 3, 3]).to(image_tensor.device)

    # Edge detection using the Sobel filter
    edges_x = F.conv2d(image_tensor.unsqueeze(0), sobel_x, padding=1)
    edges_y = F.conv2d(image_tensor.unsqueeze(0), sobel_y, padding=1)
    edges = torch.sqrt(edges_x**2 + edges_y**2 + 1e-7).squeeze(0)
    return edges

def apply_laplacian_filter(image_tensor):
    """ Apply Laplacian filter to detect edges in the image. """
    # Laplacian kernel
    laplacian_kernel = torch.tensor([[ 0., 1., 0.],
                                     [ 1., -4., 1.],
                                     [ 0., 1., 0.]], dtype=torch.float32).view(1, 1, 3, 3).expand([3, 3, 3, 3]).to(image_tensor.device)

    # Edge detection using the Laplacian filter
    edges = F.conv2d(image_tensor.unsqueeze(0), laplacian_kernel, padding=1).squeeze(0)
    return edges

def edge_loss(img, gt, mode='sobel'):
    edge_func = apply_sobel_filter if mode == 'sobel' else apply_laplacian_filter
    img_edge, gt_edge = edge_func(img), edge_func(gt)
    loss = (img_edge-gt_edge).square().mean()
    return loss
