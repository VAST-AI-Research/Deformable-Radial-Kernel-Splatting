"""Compute the scene center as the closest point to all camera optical-axis rays
(least squares), and the radius as the mean camera-to-center distance.

center = argmin_c  sum_i || (I - d_i d_i^T)(c - o_i) ||^2   (point nearest all rays)
radius = mean_i || c - o_i ||
Inner skybox shell is placed at 2 * radius.
"""
import argparse, json, os, sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from argparse import Namespace
from scene import Scene
from scene.gaussian_model import DRKModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source-path", required=True)
    ap.add_argument("-m", "--model-path", default=None)
    ap.add_argument("--unit-meter", type=float, default=8.76)
    ap.add_argument("--shell-mult", type=float, default=2.0, help="inner shell radius = mult * radius")
    ap.add_argument("--n-shells", type=int, default=6)
    ap.add_argument("--shell-growth", type=float, default=1.6, help="geometric ratio between shells")
    args = ap.parse_args()

    ds = Namespace(source_path=os.path.abspath(args.source_path),
                   model_path=os.path.abspath(args.model_path or args.source_path),
                   images="images", resolution=-1, white_background=False, data_device="cpu",
                   eval=True, val_as_train=False, gs_type="DRK", metric_masked=False)
    model = DRKModel(3, kernel_K=8)
    scene = Scene(ds, model, load_iteration=-1, shuffle=False, resolution_scales=[1.0]) if args.model_path \
        else Scene(ds, model, shuffle=False, resolution_scales=[1.0])
    cams = scene.getTrainCameras() + scene.getTestCameras()

    os_, ds_ = [], []
    for cam in cams:
        o = cam.camera_center.detach().cpu().numpy().reshape(3)
        # optical axis (world): camera looks along +z in camera space; world dir = R_c2w @ [0,0,1].
        # world_view_transform is world->cam (row-vector, stored transposed). Its rotation block's
        # 3rd ROW maps to the world +z axis of the camera -> use it as the forward direction.
        wv = cam.world_view_transform.detach().cpu().numpy()  # 4x4
        Rt = wv[:3, :3]            # rotation part (world->cam, transposed-stored)
        d = Rt[:, 2]               # camera forward in world (validated below by center sanity)
        d = d / (np.linalg.norm(d) + 1e-9)
        os_.append(o); ds_.append(d)
    os_ = np.stack(os_); ds_ = np.stack(ds_)

    # closest point to the set of rays
    def solve(dirs):
        A = np.zeros((3, 3)); b = np.zeros(3)
        for o, d in zip(os_, dirs):
            P = np.eye(3) - np.outer(d, d)
            A += P; b += P @ o
        return np.linalg.solve(A, b)

    c1 = solve(ds_)
    c2 = solve(-ds_)  # forward sign ambiguity guard; pick the center the cameras face toward
    # the true center should be IN FRONT of most cameras (dot(c-o, d) > 0)
    def front_frac(c, dirs):
        v = c[None] - os_; v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        return float((np.sum(v * dirs, axis=1) > 0).mean())
    center, dirs = (c1, ds_) if front_frac(c1, ds_) >= front_frac(c2, -ds_) else (c2, -ds_)

    cam_dists = np.linalg.norm(os_ - center[None], axis=1)
    radius = float(cam_dists.mean())
    inner = args.shell_mult * radius
    shells_units = [inner * (args.shell_growth ** k) for k in range(args.n_shells)]
    shells_m = [r * args.unit_meter for r in shells_units]

    out = {
        "n_cameras": len(cams),
        "center": center.tolist(),
        "radius_units": radius,
        "radius_m": radius * args.unit_meter,
        "cam_dist_min_max": [float(cam_dists.min()), float(cam_dists.max())],
        "front_facing_fraction": front_frac(center, dirs),
        "inner_shell_units": inner,
        "shells_units": shells_units,
        "shells_m": ",".join(f"{m:.1f}" for m in shells_m),
        "shells_m_str_for_train_background": ",".join(f"{m:.0f}" for m in shells_m),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
