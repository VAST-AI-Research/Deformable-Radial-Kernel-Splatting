import math
import torch
import colorsys
import numpy as np
from gui_utils.drk_gui import DRKUI

def generate_quaternions(K):
    quaternions = []
    for n in range(K):  # Step by 2 for even rotations
        theta = math.pi * n / K
        q = (
            math.sin(theta / 2),
            math.cos(theta / 2),
            0,
            0,
        )
        quaternions.append(q)
    return quaternions


def generate_distinct_colors(K):
    colors = []
    for i in range(K):
        # Use HSV space to distribute colors evenly, then convert to RGB
        hue = i / K
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append([r, g, b])
    return colors

K = 8
single_surfel = False
gui_mode = True

if single_surfel:
    xyz = torch.zeros([1, 3]).float().cuda()
    l1l2_rate = torch.zeros([1, 1]).cuda()
    scale = torch.zeros([1, K]).cuda()
    theta = torch.zeros([1, K]).cuda()
    acutance = 1 * torch.ones([1, 1]).cuda()
    color = torch.tensor([[170., 178., 255.]]).cuda() / 255.
    rotation = torch.tensor([[2**.5/2, 2**.5/2, 0., 0.]]).cuda()
else:
    drk_num = 16
    xyz = torch.zeros([drk_num, 3]).float().cuda()
    l1l2_rate = torch.zeros([drk_num, 1]).cuda()
    scale = torch.zeros([drk_num, K]).cuda()
    theta = torch.zeros([drk_num, K]).cuda()
    acutance = 5 * torch.ones([drk_num, 1]).cuda()
    color = torch.tensor(generate_distinct_colors(drk_num)).cuda()
    rotation = torch.tensor(generate_quaternions(drk_num)).cuda()

W, H = 800, 800
fov = 90 * torch.pi / 180
focal = W / 2 / np.tan(fov / 2)

pp = torch.tensor((W/2, H/2)).cuda()
intr = torch.tensor([focal, focal, pp[0], pp[1]]).cuda().float()

gui = DRKUI(xyz=xyz, scale=scale, rotation=rotation, color=color, intr=intr, W=W, H=H, theta=theta, acutance=acutance, l1l2_rate=l1l2_rate, radius=3, white_background=True, gui_mode=gui_mode)
gui.render()

