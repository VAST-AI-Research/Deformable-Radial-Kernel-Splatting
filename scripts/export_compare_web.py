"""Export a single WebGL2 viewer that toggles between our DRK soup and a 3DGS soup
(same pipeline) so the quality/FPS difference can be felt live on a laptop.

Bundles: drk.bin [N,104] + drk_means.bin, gs.bin [M,58] + gs_means.bin, manifest.json, index.html.
Serve over http (or use --embed for a double-click single file).  WebGL2 = the mobile GLES path.
"""
import argparse, json, os, shutil, sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from drk2mesh import load_drk_metadata
from scene.gaussian_model import DRKModel, GaussianModel
from scripts.drk_soup_gl import build_static, find_ckpt
from scripts.gs_soup_gl import build_gs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drk-model", required=True); ap.add_argument("--drk-iteration", type=int, default=-1)
    ap.add_argument("--gs-model", required=True); ap.add_argument("--gs-iteration", type=int, default=-1)
    ap.add_argument("--keep-gs", type=int, default=-1, help="subsample GS to top-N by opacity (-1=all)")
    ap.add_argument("--out", required=True); ap.add_argument("--tex-w", type=int, default=2048)
    ap.add_argument("--embed", action="store_true"); ap.add_argument("--name", default="DRK vs 3DGS")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    man = {"name": args.name, "tex_w": int(args.tex_w), "fovy": 0.8}

    # --- DRK ---
    ck = find_ckpt(os.path.abspath(args.drk_model), args.drk_iteration)
    meta = load_drk_metadata(ck, None, None); shd = int(meta.get("sh_degree", 3))
    dm = DRKModel(shd, kernel_K=8); dm.acutance_interval_list = [[meta["acutance_min"], meta["acutance_max"]]] * 4
    dm.load_ply(ck); dm.update(args.drk_iteration if args.drk_iteration > 0 else 35000); dm.eval()
    sdata, STRIDE, N, sh = build_static(dm, 1.0, 8)
    drk = np.concatenate([sdata, sh], axis=1).astype("f4")
    drk.tofile(os.path.join(args.out, "drk.bin"))
    np.ascontiguousarray(sdata[:, 0:3]).astype("f4").tofile(os.path.join(args.out, "drk_means.bin"))
    man["drk"] = {"n": int(N), "stride": 104, "sh_degree": shd}

    # --- GS ---
    gm = GaussianModel(3)
    gck = find_ckpt(os.path.abspath(args.gs_model), args.gs_iteration)
    gm.load_ply(gck)
    gsdata, GSTRIDE, M, gmeans = build_gs(gm, args.keep_gs)
    gsdata.tofile(os.path.join(args.out, "gs.bin"))
    np.ascontiguousarray(gsdata[:, 0:3]).astype("f4").tofile(os.path.join(args.out, "gs_means.bin"))
    man["gs"] = {"n": int(M), "stride": 58, "sh_degree": 3}

    # camera framing from DRK means (robust 50th pct)
    mn = np.ascontiguousarray(sdata[:, 0:3])
    center = np.median(mn, axis=0)
    radius = float(np.percentile(np.linalg.norm(mn - center[None], axis=1), 50))
    man["center"] = [float(x) for x in center]; man["cam_dist"] = max(radius * 2.2, 1.0)

    json.dump(man, open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
    html = open(os.path.join(REPO, "scripts", "web", "compare.html")).read()
    open(os.path.join(args.out, "index.html"), "w").write(html)

    if args.embed:
        import base64
        def b64(b): return base64.b64encode(b).decode("ascii")
        E = {"manifest": man,
             "drk": b64(drk.tobytes()), "drk_means": b64(np.ascontiguousarray(sdata[:, 0:3]).astype("f4").tobytes()),
             "gs": b64(gsdata.tobytes()), "gs_means": b64(np.ascontiguousarray(gsdata[:, 0:3]).astype("f4").tobytes())}
        inj = "<script>window.EMBED=" + json.dumps(E) + ";</script>\n<script>"
        open(os.path.join(args.out, "index_standalone.html"), "w").write(html.replace("<script>", inj, 1))
        print("standalone:", os.path.join(args.out, "index_standalone.html"))
    print(f"DRK {N} prims + 3DGS {M} prims -> {args.out}  (serve over http or use index_standalone.html)")


if __name__ == "__main__":
    main()
