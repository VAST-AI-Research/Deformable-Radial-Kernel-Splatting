"""Export a DRK model as a self-contained WebGL2 triangle-soup viewer folder.

Produces <out>/ : index.html (viewer), soup.bin ([N,104] f32 per-primitive data),
means.bin ([N,3] f32 for per-frame depth sort), manifest.json, skybox/*.png (optional).

Open it by serving the folder over http (browsers block file:// fetch):
    cd <out> && python3 -m http.server 8000   # then open http://localhost:8000

WebGL2 is the mobile/GLES rendering path, so the in-browser FPS is a real mobile-feasibility proxy.
"""
import argparse, json, os, shutil, sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from drk2mesh import load_drk_metadata
from scene.gaussian_model import DRKModel
from scripts.drk_soup_gl import build_static, find_ckpt   # reuse the exact, validated packing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model-path", required=True)
    ap.add_argument("--load-iteration", type=int, default=-1)
    ap.add_argument("--kernel-K", type=int, default=8)
    ap.add_argument("--cover", type=float, default=1.0)
    ap.add_argument("--skybox", default=None, help="skybox asset dir to bundle (bg_meta + shell*_rgb/alpha.png + sky.png)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tex-w", type=int, default=2048, help="data-texture width in texels")
    ap.add_argument("--embed", action="store_true", help="also write index_standalone.html with data inlined (double-click, no server)")
    args = ap.parse_args()

    ckpt = find_ckpt(os.path.abspath(args.model_path), args.load_iteration)
    if not ckpt:
        raise SystemExit("no checkpoint found")
    meta = load_drk_metadata(ckpt, None, None)
    sh_degree = int(meta.get("sh_degree", 3))
    model = DRKModel(sh_degree, kernel_K=args.kernel_K)
    amn, amx = meta["acutance_min"], meta["acutance_max"]
    model.acutance_interval_list = [[amn, amx]] * 4
    model.load_ply(ckpt)
    it = args.load_iteration if args.load_iteration > 0 else 35000
    model.update(it)
    model.eval()

    sdata, STRIDE, N, sh = build_static(model, args.cover, args.kernel_K)  # [N,56], [N,48]
    assert STRIDE == 56
    data = np.concatenate([sdata, sh], axis=1).astype("f4")               # [N,104]
    means = np.ascontiguousarray(sdata[:, 0:3]).astype("f4")

    os.makedirs(args.out, exist_ok=True)
    data.tofile(os.path.join(args.out, "soup.bin"))
    means.tofile(os.path.join(args.out, "means.bin"))

    bmeta = json.load(open(os.path.join(args.skybox, "bg_meta.json"))) if args.skybox else None
    # default camera framing: prefer the ray-convergence center + scene radius (skybox meta);
    # else a robust (50th-pct) estimate from the primitive cloud (avoids far-floater contamination).
    if bmeta is not None:
        center = np.array(bmeta["center"], dtype="f4")
        radius = float(bmeta["radii_units"][0]) * 0.5   # inner shell = 2*radius
    else:
        center = np.median(means, axis=0)
        radius = float(np.percentile(np.linalg.norm(means - center[None], axis=1), 50))
    cam_dist = max(radius * 2.2, 1.0)

    manifest = {
        "name": args.name or os.path.basename(os.path.normpath(args.model_path)),
        "n": int(N), "stride": 104, "tex_w": int(args.tex_w), "sh_degree": int(sh_degree),
        "center": [float(x) for x in center], "cam_dist": float(cam_dist), "fovy": 0.8,
    }

    if args.skybox:
        from PIL import Image
        sbdir = os.path.join(args.out, "skybox")
        os.makedirs(sbdir, exist_ok=True)
        radii = [float(r) for r in bmeta["radii_units"]]
        for k in range(len(radii)):
            rgb = Image.open(os.path.join(args.skybox, f"shell{k}_rgb.png")).convert("RGB")
            al = Image.open(os.path.join(args.skybox, f"shell{k}_alpha.png")).convert("L")
            rgba = Image.merge("RGBA", (*rgb.split(), al))
            rgba.save(os.path.join(sbdir, f"shell{k}_rgba.png"))
        shutil.copy(os.path.join(args.skybox, "sky.png"), os.path.join(sbdir, "sky.png"))
        manifest["skybox"] = {"center": [float(x) for x in bmeta["center"]], "radii": radii}

    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
    viewer_html = open(os.path.join(REPO, "scripts", "web", "viewer.html")).read()
    open(os.path.join(args.out, "index.html"), "w").write(viewer_html)

    if args.embed:
        import base64
        def b64(b):
            return base64.b64encode(b).decode("ascii")
        embed = {"manifest": manifest,
                 "soup": b64(data.tobytes()),
                 "means": b64(means.tobytes())}
        if args.skybox:
            embed["shells"] = ["data:image/png;base64," + b64(open(os.path.join(args.out, "skybox", f"shell{k}_rgba.png"), "rb").read())
                               for k in range(len(manifest["skybox"]["radii"]))]
            embed["sky"] = "data:image/png;base64," + b64(open(os.path.join(args.out, "skybox", "sky.png"), "rb").read())
        inject = "<script>window.EMBED=" + json.dumps(embed) + ";</script>\n<script>"
        standalone = viewer_html.replace("<script>", inject, 1)  # inline before the main script
        sp = os.path.join(args.out, "index_standalone.html")
        open(sp, "w").write(standalone)
        print(f"  standalone single-file (double-click, no server): {sp} ({len(standalone)/1e6:.0f} MB)")

    size_mb = (data.nbytes + means.nbytes) / 1e6
    print(f"exported {N} primitives ({size_mb:.1f} MB data) -> {args.out}")
    print(f"  data-texture: {args.tex_w} x {int(np.ceil(N*26/args.tex_w))} texels (RGBA32F)")
    print("  VIEW:  cd %s && python3 -m http.server 8000   # open http://localhost:8000" % args.out)


if __name__ == "__main__":
    main()
