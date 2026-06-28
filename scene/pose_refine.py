"""In-training camera pose refinement (a learned bundle adjustment).

VGGT-Omega gives feed-forward poses that are NOT globally bundle-adjusted, which
caps multi-view reconstruction (~17 dB even for 3DGS). COLMAP is unavailable in
this environment, so we refine poses jointly with the model instead.

Trick (no CUDA change): for camera c with a learnable SE3 delta D_c, render with
the FIXED base camera but rigidly transform all primitives by D_c. Moving the
world by D_c is equivalent to moving the camera by D_c^-1, and gradients flow to
D_c through the rasterizer's d(loss)/d(means3D) and d(loss)/d(rotations).
"""
import torch
import torch.nn as nn


def axis_angle_to_matrix(r):
    """r: [N,3] axis-angle -> [N,3,3] rotation via the so(3) exponential.

    R = I + A·[r]x + B·[r]x^2 with A = sin(t)/t, B = (1-cos t)/t^2 (t=|r|).
    Uses Taylor series for small t so it is finite AND differentiable at r=0
    (a naive Rodrigues normalizes by t and loses the rotation gradient at init)."""
    eps = 1e-5
    t2 = (r * r).sum(dim=1)                            # [N]
    # sqrt(t2+eps^2) is smooth with finite gradient at r=0 (avoids the 0*inf NaN
    # from sqrt(0) that a where()-masked Rodrigues branch would produce).
    t = torch.sqrt(t2 + eps * eps)
    A = torch.sin(t) / t                               # ->1 as t->0
    B = (1.0 - torch.cos(t)) / (t * t)                 # ->0.5 as t->0
    rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
    zero = torch.zeros_like(rx)
    W = torch.stack([zero, -rz, ry, rz, zero, -rx, -ry, rx, zero], dim=1).view(-1, 3, 3)
    I = torch.eye(3, device=r.device, dtype=r.dtype).expand(r.shape[0], 3, 3)
    A = A.view(-1, 1, 1); B = B.view(-1, 1, 1)
    return I + A * W + B * torch.bmm(W, W)


class PoseRefine(nn.Module):
    def __init__(self, num_cameras):
        super().__init__()
        self.dr = nn.Parameter(torch.zeros(num_cameras, 3))   # axis-angle delta
        self.dt = nn.Parameter(torch.zeros(num_cameras, 3))   # translation delta

    def transform(self, idx, xyz, rot9):
        """Apply camera idx's SE3 delta to primitive centers (xyz [P,3]) and
        kernel frames (rot9 [P,9], row-major 3x3). Differentiable in dr/dt."""
        R = axis_angle_to_matrix(self.dr[idx:idx + 1])[0]      # [3,3]
        t = self.dt[idx]                                       # [3]
        xyz2 = xyz @ R.transpose(0, 1) + t
        rotm = rot9.view(-1, 3, 3)
        rot2 = torch.matmul(R.unsqueeze(0), rotm).reshape(-1, 9)
        return xyz2, rot2

    @torch.no_grad()
    def magnitude(self):
        return float(self.dr.norm(dim=1).mean()), float(self.dt.norm(dim=1).mean())
