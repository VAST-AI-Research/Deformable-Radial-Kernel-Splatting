"""Track A: DRK 'triangle soup' rendered by a TRADITIONAL rasterization pipeline.

Each DRK primitive becomes one instanced quad lying in the primitive's plane; a GLSL
fragment shader evaluates the exact DRK radial kernel (ported from forward.cu's general
soft path) to produce per-fragment alpha. Primitives are depth-sorted per-frame
(back-to-front) and composited single-pass with standard "over" alpha blending.

This reproduces the soft DRK appearance (~22 dB) using a hardware rasterizer + a
per-primitive sort instead of the CUDA tile rasterizer with per-pixel cache_sort, i.e.
"triangle soup at much higher FPS" -- the headline deliverable.

Renders the test split, reports mesh(soup)-vs-GT PSNR/SSIM/L1 and FPS, and (optionally)
the reference soft-DRK PSNR for the same model so the conversion gap is visible.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from argparse import Namespace
from drk2mesh import load_drk_metadata
from gaussian_renderer import default_pipe
from scene import Scene
from scene.gaussian_model import DRKModel
from utils.image_utils import psnr as psnr_fn
from utils.loss_utils import ssim as ssim_fn

import moderngl

KERNEL_K = 8

# Geometry: per-primitive K-gon FAN (K triangles), gl_VertexID-driven (no vertex buffer).
# Each fan vertex sits at kernel_vecs[idx] * support, where support tightly bounds the
# kernel to its alpha=1/255 contour per-primitive -> minimal overdraw.
VERT_SRC = """
#version 430
layout(std430, binding=0) buffer Static { float sdata[];  };  // N * STRIDE
layout(std430, binding=2) buffer Order  { int   order[];  };  // draw order (back->front)
layout(std430, binding=3) buffer SH     { float shdata[]; };  // N * 48 (16 bases x rgb)
uniform mat4 u_mvp;
uniform int u_flip_y;
uniform int u_stride;
uniform float u_cover;
uniform vec3 u_campos;
uniform vec2 u_res;
uniform int u_sh_degree;
in float a_corner;        // indexed-fan corner id: 0=center, 1..K=boundary vertex
out vec2 v_uv;
flat out int v_prim;
out vec3 v_rgb;
flat out vec2 v_center;   // screen-space primitive center (window px) for low-pass
flat out float v_invcos;  // 1/|cos(view dir, normal)| for low-pass
vec3 shc(int prim, int idx) {
    int o = prim*48 + idx*3;
    return vec3(shdata[o], shdata[o+1], shdata[o+2]);
}
vec3 eval_sh_color(int prim, vec3 d) {
    float x=d.x, y=d.y, z=d.z;
    float xx=x*x, yy=y*y, zz=z*z, xy=x*y, yz=y*z, xz=x*z;
    vec3 r = 0.28209479177387814 * shc(prim,0);
    if (u_sh_degree > 0) r += -0.4886025119029199*y*shc(prim,1) + 0.4886025119029199*z*shc(prim,2) - 0.4886025119029199*x*shc(prim,3);
    if (u_sh_degree > 1) r += 1.0925484305920792*xy*shc(prim,4) - 1.0925484305920792*yz*shc(prim,5)
        + 0.31539156525252005*(2.0*zz-xx-yy)*shc(prim,6) - 1.0925484305920792*xz*shc(prim,7) + 0.5462742152960396*(xx-yy)*shc(prim,8);
    if (u_sh_degree > 2) r += -0.5900435899266435*y*(3.0*xx-yy)*shc(prim,9) + 2.890611442640554*xy*z*shc(prim,10)
        - 0.4570457994644658*y*(4.0*zz-xx-yy)*shc(prim,11) + 0.3731763325901154*z*(2.0*zz-3.0*xx-3.0*yy)*shc(prim,12)
        - 0.4570457994644658*x*(4.0*zz-xx-yy)*shc(prim,13) + 1.445305721320277*z*(xx-yy)*shc(prim,14) - 0.5900435899266435*x*(xx-3.0*yy)*shc(prim,15);
    return max(r + vec3(0.5), vec3(0.0));
}
void main() {
    int prim = order[gl_InstanceID];
    int b = prim * u_stride;
    vec3 mean = vec3(sdata[b+0], sdata[b+1], sdata[b+2]);
    vec3 uax  = vec3(sdata[b+3], sdata[b+4], sdata[b+5]);
    vec3 vax  = vec3(sdata[b+6], sdata[b+7], sdata[b+8]);
    vec3 nrm  = vec3(sdata[b+53], sdata[b+54], sdata[b+55]);
    float opacity = sdata[b+10];
    // per-primitive support: kernel alpha hits 1/255 at uv_norm = ln(255*opacity);
    // uv_norm ~ 0.5*d^2 along a kernel_vec of magnitude=scale -> d = sqrt(2*ln(255*opa)).
    float sup = u_cover * clamp(sqrt(max(2.0*log(255.0*opacity + 1.0), 0.04)), 0.5, 3.5);
    int kvb = b + 21;
    int ci = int(a_corner + 0.5);   // 0 = center, 1..K = boundary vertex (indexed fan)
    vec2 uv = (ci == 0) ? vec2(0.0) : vec2(sdata[kvb + 2*(ci-1)], sdata[kvb + 2*(ci-1) + 1]) * sup;
    vec3 world = mean + uv.x * uax + uv.y * vax;
    vec4 clip = u_mvp * vec4(world, 1.0);
    if (u_flip_y != 0) clip.y = -clip.y;
    gl_Position = clip;
    v_uv = uv;
    v_prim = prim;
    // low-pass support data (per-primitive, view dependent)
    vec3 op = mean - u_campos;
    float opn = dot(op, nrm);
    v_invcos = length(op) / max(abs(opn), 1e-7);
    v_rgb = eval_sh_color(prim, normalize(op));   // view-dependent SH color, on GPU
    vec4 cc = u_mvp * vec4(mean, 1.0);
    if (u_flip_y != 0) cc.y = -cc.y;
    vec2 ndc = cc.xy / cc.w;
    v_center = (ndc * 0.5 + 0.5) * u_res;   // window-pixel coords (matches gl_FragCoord)
}
"""

# Fragment: exact DRK general soft-path kernel (forward.cu:756-863), low-pass omitted.
FRAG_SRC = """
#version 430
layout(std430, binding=0) buffer Static { float sdata[]; };
uniform int u_stride;
uniform float u_lpf;     // FilterInvSquare, scaled for supersampling
uniform int u_mode;      // 0=DRK kernel, 1=flat (per-prim const alpha), 2=opaque
in vec2 v_uv;
flat in int v_prim;
in vec3 v_rgb;
flat in vec2 v_center;
flat in float v_invcos;
out vec4 frag;
const float PI = 3.14159265359;
void main() {
    int b = v_prim * u_stride;
    float opacity  = sdata[b+10];
    float k_acu    = sdata[b+11];
    float l1l2     = sdata[b+12];
    if (u_mode > 0) {  // flat (const per-prim alpha) / opaque triangles -> isolates fragment-kernel cost
        float a = (u_mode==2) ? 1.0 : opacity;
        if (a < 1.0/255.0) discard;
        frag = vec4(max(v_rgb, vec3(0.0)), min(0.99, a)); return;
    }
    // arrays
    int tb  = b + 13;        // thetas[8]
    int kvb = b + 21;        // kernel_vecs[8] -> 16 floats (x,y)
    int sib = b + 37;        // scale_inv2[8]
    int idb = b + 45;        // inv_delta[8]
    vec2 uv = v_uv;
    // Step 2: angular segment selection
    float theta = atan(uv.y, uv.x);
    if (theta < 0.0) theta += 2.0*PI;
    theta = min(theta * (0.5/PI), 1.0);
    int k = 0;
    for (int i = 0; i < KERNEL_K-1; ++i) k += (sdata[tb+i] < theta) ? 1 : 0;
    float uv_l2norm = max(dot(uv,uv), 1e-8);
    float theta_l = (k==0) ? 0.0 : sdata[tb+k-1];
    float theta_r = sdata[tb+k];
    float linear_rate = (theta - theta_l) / (theta_r - theta_l);
    // smoothstep S-curve replaces 0.5*(cos((1-lr)*PI)+1): same 0/0.5/1 anchors, no transcendental
    float rate = linear_rate * linear_rate * (3.0 - 2.0 * linear_rate);
    float inv_sl2 = sdata[sib+k];
    float inv_sr2 = (k==KERNEL_K-1) ? sdata[sib+0] : sdata[sib+k+1];
    // Step 3: affine map into segment frame
    float inv_delta = sdata[idb+k];
    vec2 e1 = vec2(sdata[kvb+2*k],   sdata[kvb+2*k+1]);
    int k2 = (k==KERNEL_K-1) ? 0 : k+1;
    vec2 e2 = vec2(sdata[kvb+2*k2],  sdata[kvb+2*k2+1]);
    vec2 uv_t = vec2(( e2.y*uv.x - e2.x*uv.y) * inv_delta,
                     (-e1.y*uv.x + e1.x*uv.y) * inv_delta);
    // Step 4: L1/L2 blended norm
    float l1 = abs(uv_t.x) + abs(uv_t.y);
    l1 = max(l1*l1*0.5, 1e-8);
    float l2 = max(uv_l2norm * (rate*inv_sr2 + (1.0-rate)*inv_sl2) * 0.5, 1e-8);
    float uv_norm = l1l2*l1 + (1.0-l1l2)*l2;
    // Step 5: gaussian + acutance sharpen
    float ko = exp(-uv_norm);
    float alpha;
    if (k_acu >= 1.0 - 1e-6) {
        alpha = (ko >= 0.5 ? 1.0 : 0.0) * opacity;
    } else {
        float a = min(k_acu, 0.999999);
        float s;
        if (ko < (1.0+a)/4.0)      s = (1.0-a)/(1.0+a)*ko;
        else                        s = (1.0+a)/(1.0-a)*ko - a/(1.0-a);
        if (ko >= (3.0-a)/4.0)      s = (1.0-a)/(1.0+a)*ko + 2.0*a/(1.0+a);
        alpha = s * opacity;
    }
    // Step 6: low-pass floor (forward.cu) -> min ~1px footprint / anti-alias distant prims
    vec2 r2 = (v_center - gl_FragCoord.xy) * v_invcos;
    float Glps = exp(-0.5 * u_lpf * dot(r2, r2));
    alpha = max(alpha, opacity * Glps);
    if (alpha < 1.0/255.0) discard;
    alpha = min(0.99, alpha);
    frag = vec4(max(v_rgb, vec3(0.0)), alpha);
}
"""


# --- Skybox background pass: composite multi-shell equirect panorama behind the soup ---
BG_VERT = """
#version 430
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""
BG_FRAG = """
#version 430
uniform mat4 u_finv;        // inverse(full_proj), uploaded like u_mvp (GL reads it transposed)
uniform vec3 u_campos;
uniform vec3 u_center;
uniform vec2 u_res;
uniform int u_nshells;
uniform float u_radii[8];
uniform sampler2D u_shell[8];
uniform sampler2D u_sky;
out vec4 frag;
const float PI = 3.14159265359;
vec2 dir2uv(vec3 d) { return vec2(atan(d.x, d.z)/(2.0*PI)+0.5, acos(clamp(d.y,-1.0,1.0))/PI); }
void main() {
    vec2 ndc = (gl_FragCoord.xy / u_res) * 2.0 - 1.0;
    vec4 wh = u_finv * vec4(ndc, 1.0, 1.0);
    vec3 pfar = wh.xyz / wh.w;
    vec3 d = normalize(pfar - u_campos);
    vec3 o = u_campos - u_center;
    float od = dot(o, d), oo = dot(o, o);
    vec3 C = vec3(0.0); float T = 1.0;
    for (int k = 0; k < u_nshells; ++k) {
        float r = u_radii[k];
        float disc = od*od - oo + r*r;
        if (disc > 0.0) {
            float t = -od + sqrt(disc);
            vec3 dk = normalize(o + t*d);
            vec4 rgba = texture(u_shell[k], dir2uv(dk));
            float a = rgba.a;
            C += T * a * rgba.rgb; T *= (1.0 - a);
        }
    }
    C += T * texture(u_sky, dir2uv(d)).rgb;
    frag = vec4(C, 1.0);
}
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--source-path", required=True)
    p.add_argument("-m", "--model-path", required=True)
    p.add_argument("--load-iteration", type=int, default=-1)
    p.add_argument("--split", choices=["train", "test", "all"], default="test")
    p.add_argument("--views", type=int, default=-1)
    p.add_argument("--kernel-K", type=int, default=8)
    p.add_argument("--cover", type=float, default=1.0, help="global multiplier on per-primitive kernel support (1.0 = tight)")
    p.add_argument("--supersample", type=int, default=1)
    p.add_argument("--gl-device", type=int, default=0, help="EGL/physical GPU index for the GL context (use a free GPU)")
    p.add_argument("--frag-mode", choices=["kernel", "flat", "opaque"], default="kernel",
                   help="kernel=DRK soft soup; flat=const per-prim alpha (no kernel); opaque=z-buffered triangles")
    p.add_argument("--render-sh-degree", type=int, default=-1)
    p.add_argument("--bench-iters", type=int, default=0, help="extra timed frames for FPS (per view)")
    p.add_argument("--save-images", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--skybox", default=None, help="skybox asset dir (bg_meta.json + shell*_rgb/alpha.png + sky.png) to composite behind the soup")
    p.add_argument("--video", default=None, help="write an EGL soup fly-through mp4 to this path")
    p.add_argument("--video-split", choices=["train", "test", "all"], default="all")
    p.add_argument("--video-fps", type=int, default=24)
    p.add_argument("--video-mode", choices=["sidebyside", "soup"], default="sidebyside")
    return p.parse_args()


def make_dataset_args(args):
    return Namespace(source_path=os.path.abspath(args.source_path), model_path=os.path.abspath(args.model_path),
                     images="images", resolution=-1, white_background=False, data_device="cuda", eval=True,
                     val_as_train=False, gs_type="DRK", metric_masked=False)


def find_ckpt(model_path, it):
    if it and it > 0:
        pth = os.path.join(model_path, "point_cloud", f"iteration_{it}", "point_cloud.ply")
        return pth if os.path.exists(pth) else None
    root = os.path.join(model_path, "point_cloud")
    best = None
    for n in os.listdir(root):
        if n.startswith("iteration_"):
            try: i = int(n.split("_")[1])
            except: continue
            pth = os.path.join(root, n, "point_cloud.ply")
            if os.path.exists(pth): best = max(best, (i, pth)) if best else (i, pth)
    return best[1] if best else None


@torch.no_grad()
def build_static(model, cover, K):
    xyz = model.get_xyz                                  # [N,3]
    scales = model.get_scaling                           # [N,K]
    thetas = model.get_thetas                            # [N,K] in [0,1], last=1
    acut = model.get_acutance.reshape(-1)                # [N]
    l1l2 = model.get_l1l2rates.reshape(-1)               # [N]
    opacity = model.get_opacity.reshape(-1)              # [N]
    R = model.get_rotation.reshape(-1, 3, 3)             # [N,3,3] row-major (cols = kernel axes)
    N = xyz.shape[0]
    dev = xyz.device
    # kernel_vecs[ii] = (cos,sin)(angle)*scales[ii], angle = (ii==0?0:thetas[ii-1])*2pi
    ang = torch.zeros((N, K), device=dev)
    ang[:, 1:] = thetas[:, :K-1]
    ang = ang * (2.0 * np.pi)
    kv = torch.stack([torch.cos(ang) * scales, torch.sin(ang) * scales], dim=-1)  # [N,K,2]
    scale_inv2 = 1.0 / (scales * scales)                                          # [N,K]
    e1 = kv
    e2 = torch.roll(kv, shifts=-1, dims=1)
    delta = e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]
    delta = torch.sign(delta) * torch.clamp(delta.abs(), min=1e-7)
    inv_delta = 1.0 / delta                                                        # [N,K]
    uax = R[:, :, 0]  # col0
    vax = R[:, :, 1]  # col1
    half = cover * scales.max(dim=1).values                                        # [N]
    STRIDE = 56
    s = torch.zeros((N, STRIDE), device=dev)
    s[:, 0:3] = xyz
    s[:, 3:6] = uax
    s[:, 6:9] = vax
    s[:, 9] = half
    s[:, 10] = opacity
    s[:, 11] = acut
    s[:, 12] = l1l2
    s[:, 13:21] = thetas
    s[:, 21:37] = kv.reshape(N, 2 * K)
    s[:, 37:45] = scale_inv2
    s[:, 45:53] = inv_delta
    s[:, 53:56] = R[:, :, 2]   # kernel normal (col2), for low-pass
    sh = model.get_features.reshape(N, -1).contiguous().cpu().numpy().astype("f4")  # [N, 48] basis-major rgb
    return s.contiguous().cpu().numpy().astype("f4"), STRIDE, N, sh


def img_metrics(pred, tgt):
    l1 = torch.abs(pred - tgt).mean().item()
    return {"l1": l1, "psnr": psnr_fn(pred[None], tgt[None]).mean().item(),
            "ssim": ssim_fn(pred, tgt).item()}


def main():
    args = parse_args()
    K = args.kernel_K
    ckpt = find_ckpt(os.path.abspath(args.model_path), args.load_iteration)
    meta = load_drk_metadata(ckpt, None, None) if ckpt else None
    sh_degree = int(meta.get("sh_degree", 3)) if meta else 3
    render_sh = min(sh_degree, 4) if args.render_sh_degree < 0 else min(args.render_sh_degree, sh_degree, 4)
    model = DRKModel(sh_degree, kernel_K=K)
    if meta:
        amn, amx = meta["acutance_min"], meta["acutance_max"]
        model.acutance_interval_list = [[amn, amx]] * 4
    model.cache_sort = False
    model.tile_culling = False
    scene = Scene(make_dataset_args(args), model, load_iteration=args.load_iteration, shuffle=False)
    stage_iter = scene.loaded_iter or args.load_iteration or 35000
    model.update(stage_iter)
    model.eval()

    cams = (scene.getTestCameras() if args.split == "test" else
            scene.getTrainCameras() if args.split == "train" else
            scene.getTrainCameras() + scene.getTestCameras())
    if len(cams) == 0:
        cams = scene.getTrainCameras()
    if args.views > 0:
        cams = cams[:args.views]
    W = int(cams[0].image_width); H = int(cams[0].image_height)
    ss = args.supersample
    rW, rH = W * ss, H * ss

    # ---- GL setup ----
    ctx = moderngl.create_context(standalone=True, backend="egl", require=430, device_index=int(args.gl_device))
    prog = ctx.program(vertex_shader=VERT_SRC.replace("KERNEL_K", str(K)),
                       fragment_shader=FRAG_SRC.replace("KERNEL_K", str(K)))
    sdata, STRIDE, N, shdata = build_static(model, args.cover, K)
    prog["u_stride"] = STRIDE
    prog["u_flip_y"] = 0   # we flip rows after readback instead
    prog["u_cover"] = float(args.cover)
    prog["u_res"] = (float(rW), float(rH))
    prog["u_lpf"] = 16.0 / float(ss * ss)   # FilterInvSquare, scaled for supersampling
    prog["u_sh_degree"] = int(render_sh)
    static_buf = ctx.buffer(sdata.tobytes())
    sh_buf = ctx.buffer(shdata.tobytes())
    order_buf = ctx.buffer(reserve=N * 4, dynamic=True)
    # indexed K-gon fan: 1+K unique verts/prim, K triangles -> K+1 vertex-shader runs/instance
    # instead of 3K (no vertex reuse). corner id 0=center, 1..K=boundary.
    corner_np = np.arange(K + 1, dtype="f4")
    fan_idx = []
    for i in range(1, K + 1):
        fan_idx += [0, i, (i % K) + 1]
    corner_buf = ctx.buffer(corner_np.tobytes())
    idx_buf = ctx.buffer(np.array(fan_idx, dtype="i4").tobytes())
    vao = ctx.vertex_array(prog, [(corner_buf, "1f4", "a_corner")], index_buffer=idx_buf)
    # 16F framebuffer: ~2x less blend/readback bandwidth than f4, enough precision for the
    # deep low-alpha blend (8-bit lost ~2.5 dB; f2 keeps it).
    fbo = ctx.framebuffer(color_attachments=[ctx.texture((rW, rH), 4, dtype="f2")],
                          depth_attachment=ctx.depth_texture((rW, rH)))
    FRAG_MODE = {"kernel": 0, "flat": 1, "opaque": 2}[args.frag_mode]
    prog["u_mode"] = FRAG_MODE

    # ---- optional skybox (multi-shell equirect panorama composited behind the soup) ----
    bg_prog = bg_vao = bg_center = bg_radii = None
    bg_shell_tex = []
    bg_sky_tex = None
    if args.skybox:
        from PIL import Image as _Im
        meta = json.load(open(os.path.join(args.skybox, "bg_meta.json")))
        bg_center = np.array(meta["center"], dtype="f4")
        bg_radii = [float(r) for r in meta["radii_units"]]
        def _equirect_tex(arr):  # arr [H,W,C] float in [0,1]
            t = ctx.texture((arr.shape[1], arr.shape[0]), arr.shape[2], np.ascontiguousarray(arr, "f4").tobytes(), dtype="f4")
            t.repeat_x = True; t.repeat_y = False; t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            return t
        for k in range(len(bg_radii)):
            rgb = np.asarray(_Im.open(os.path.join(args.skybox, f"shell{k}_rgb.png")).convert("RGB")).astype("f4") / 255.0
            al = np.asarray(_Im.open(os.path.join(args.skybox, f"shell{k}_alpha.png")).convert("L")).astype("f4") / 255.0
            bg_shell_tex.append(_equirect_tex(np.concatenate([rgb, al[..., None]], axis=2)))
        bg_sky_tex = _equirect_tex(np.asarray(_Im.open(os.path.join(args.skybox, "sky.png")).convert("RGB")).astype("f4") / 255.0)
        bg_prog = ctx.program(vertex_shader=BG_VERT, fragment_shader=BG_FRAG)
        bg_prog["u_res"] = (float(rW), float(rH))
        bg_prog["u_center"] = tuple(bg_center.tolist())
        bg_prog["u_nshells"] = len(bg_radii)
        bg_prog["u_radii"] = tuple((bg_radii + [0.0] * 8)[:8])
        bg_prog["u_shell"] = tuple(list(range(len(bg_shell_tex))) + [0] * (8 - len(bg_shell_tex)))
        bg_prog["u_sky"] = 7
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        bg_quad_buf = ctx.buffer(quad.tobytes())
        bg_vao = ctx.vertex_array(bg_prog, [(bg_quad_buf, "2f4", "in_pos")])
        print(f"skybox loaded: {len(bg_radii)} shells {[round(r,2) for r in bg_radii]} center {bg_center.tolist()}")

    xyz_t = model.get_xyz  # [N,3] on cuda
    ones_t = torch.ones((N, 1), device=xyz_t.device)

    prof = {"sort": 0.0, "color": 0.0, "upload": 0.0, "draw": 0.0, "readback": 0.0, "n": 0}

    @torch.no_grad()
    def render(cam, record=False):
        def tick():
            torch.cuda.synchronize(); return time.time()
        t = tick()
        # frustum cull + per-primitive view-space depth for back-to-front sort
        wv = cam.world_view_transform  # [4,4], row-vector: p_view = p_world @ wv
        fp = cam.full_proj_transform   # [4,4], row-vector: clip = p_world @ fp
        homo = torch.cat([xyz_t, ones_t], dim=1)
        clip = homo @ fp
        w = clip[:, 3]
        ndc = clip[:, :3] / w.unsqueeze(1).clamp_min(1e-6)
        vis = (w > 0.01) & (ndc[:, 0].abs() < 1.3) & (ndc[:, 1].abs() < 1.3) & (ndc[:, 2] > -1.0) & (ndc[:, 2] < 1.0)
        depth_v = (homo @ wv)[:, 2]
        vis_idx = torch.nonzero(vis, as_tuple=False).squeeze(1)
        order = vis_idx[torch.argsort(depth_v[vis_idx], descending=True)].to(torch.int32).cpu().numpy()  # far->near
        n_inst = int(order.shape[0])
        t1 = tick()
        t2 = tick()   # SH color now evaluated on-GPU in the vertex shader (no per-frame upload)
        order_buf.write(order.tobytes())
        static_buf.bind_to_storage_buffer(0)
        order_buf.bind_to_storage_buffer(2)
        sh_buf.bind_to_storage_buffer(3)
        mvp = np.ascontiguousarray(cam.full_proj_transform.detach().cpu().numpy(), dtype="f4")
        campos_np = cam.camera_center.detach().cpu().numpy().astype("f4")
        prog["u_mvp"].write(mvp.tobytes())
        prog["u_campos"].write(campos_np.tobytes())
        fbo.use()
        ctx.viewport = (0, 0, rW, rH)
        ctx.clear(0.0, 0.0, 0.0, 0.0)
        ctx.disable(moderngl.DEPTH_TEST)
        t3 = time.time()
        if bg_prog is not None:  # skybox background pass (opaque, fills the framebuffer)
            finv = np.ascontiguousarray(np.linalg.inv(cam.full_proj_transform.detach().cpu().numpy()), dtype="f4")
            bg_prog["u_finv"].write(finv.tobytes())
            bg_prog["u_campos"].write(campos_np.tobytes())
            for kk, shtex in enumerate(bg_shell_tex):
                shtex.use(kk)
            bg_sky_tex.use(7)
            ctx.disable(moderngl.BLEND)
            bg_vao.render(mode=moderngl.TRIANGLE_STRIP, vertices=4)
        if FRAG_MODE == 2:  # opaque z-buffered triangles (traditional, no transparency)
            ctx.enable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        else:               # kernel soup / flat: sorted alpha blend
            ctx.enable(moderngl.BLEND); ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        vao.render(mode=moderngl.TRIANGLES, instances=n_inst)  # indexed fan (3K indices)
        ctx.finish()
        t4 = time.time()
        raw = np.frombuffer(fbo.read(components=4, dtype="f2"), dtype="float16").reshape(rH, rW, 4)
        arr = np.ascontiguousarray(raw[..., :3])  # full_proj already yields image-orient rows here
        img = torch.from_numpy(arr).to("cuda").permute(2, 0, 1).contiguous().float()  # [3,rH,rW]
        if ss > 1:
            img = torch.nn.functional.avg_pool2d(img[None], ss)[0]
        t5 = tick()
        if record:
            prof["sort"] += t1 - t; prof["color"] += t2 - t1; prof["upload"] += t3 - t2
            prof["draw"] += t4 - t3; prof["readback"] += t5 - t4; prof["n"] += 1
        return img.clamp(0, 1)

    per_view = []
    save_dir = args.save_images
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    # warmup
    _ = render(cams[0])
    t_total = 0.0; frames = 0
    for i, cam in enumerate(cams):
        torch.cuda.synchronize()
        t0 = time.time()
        img = render(cam)
        torch.cuda.synchronize()
        dt = time.time() - t0
        t_total += dt; frames += 1
        gt = cam.original_image.to("cuda").clamp(0, 1)
        m = img_metrics(img, gt)
        m["view"] = i; m["camera"] = cam.image_name; m["ms"] = dt * 1000
        per_view.append(m)
        if save_dir:
            from PIL import Image
            Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")).save(
                os.path.join(save_dir, f"soup_{i:03d}.png"))
            Image.fromarray((gt.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")).save(
                os.path.join(save_dir, f"gt_{i:03d}.png"))
    # FPS bench (steady state, no readback overhead counted separately)
    bench_ms = None
    if args.bench_iters > 0:
        cam = cams[0]
        for _ in range(5): render(cam)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(args.bench_iters): render(cam, record=True)
        torch.cuda.synchronize()
        bench_ms = (time.time() - t0) / args.bench_iters * 1000.0
        nb = max(prof["n"], 1)
        print("PROFILE ms/frame:", {k: round(prof[k] / nb * 1000, 2) for k in ("sort", "color", "upload", "draw", "readback")})

    if args.video:
        import imageio
        vsplit = (scene.getTestCameras() if args.video_split == "test" else
                  scene.getTrainCameras() if args.video_split == "train" else
                  scene.getTrainCameras() + scene.getTestCameras())
        vsplit = sorted(vsplit, key=lambda c: c.image_name)  # capture order
        frames = []
        H0, W0 = int(vsplit[0].image_height), int(vsplit[0].image_width)
        for cam in vsplit:
            img = render(cam)  # [3,h,w] in [0,1]
            soup_t = img.clamp(0, 1)
            if soup_t.shape[1:] != (H0, W0):
                soup_t = torch.nn.functional.interpolate(soup_t[None], size=(H0, W0), mode="bilinear", align_corners=False)[0]
            soup = (soup_t.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            if args.video_mode == "sidebyside":
                gt_t = cam.original_image[:3].to(soup_t.device).clamp(0, 1)
                if gt_t.shape[1:] != (H0, W0):
                    gt_t = torch.nn.functional.interpolate(gt_t[None], size=(H0, W0), mode="bilinear", align_corners=False)[0]
                gt = (gt_t.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                frame = np.concatenate([gt, soup], axis=1)
            else:
                frame = soup
            h, w = frame.shape[:2]
            frame = np.ascontiguousarray(frame[: h - (h % 2), : w - (w % 2), :3])
            frames.append(frame)
        # write PNG frames (no in-process ffmpeg -> avoids torch/OpenMP conflict), then encode
        # with the system ffmpeg in a clean subprocess env.
        import subprocess, shutil
        os.makedirs(os.path.dirname(os.path.abspath(args.video)) or ".", exist_ok=True)
        fdir = os.path.splitext(args.video)[0] + "_frames"
        os.makedirs(fdir, exist_ok=True)
        for i, fr in enumerate(frames):
            imageio.imwrite(os.path.join(fdir, f"f{i:04d}.png"), fr)
        ffmpeg = shutil.which("ffmpeg") or "/usr/local/bin/ffmpeg"
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OMP_", "KMP_", "MKL_")) and k != "LD_PRELOAD"}
        cmd = [ffmpeg, "-y", "-framerate", str(args.video_fps), "-i", os.path.join(fdir, "f%04d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", args.video]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        ok = r.returncode == 0 and os.path.exists(args.video)
        print(f"wrote {len(frames)} frames {frames[0].shape} -> {args.video if ok else fdir} "
              f"({args.video_mode}, {args.video_split}; ffmpeg {'ok' if ok else 'FAILED: '+r.stderr[-300:]})")

    mean = {k: float(np.mean([v[k] for v in per_view])) for k in ("l1", "psnr", "ssim", "ms")}
    result = {
        "model_path": os.path.abspath(args.model_path), "load_iteration": scene.loaded_iter,
        "split": args.split, "views": len(per_view), "primitives": int(N),
        "resolution": [W, H], "supersample": ss, "cover": args.cover,
        "render_sh_degree": render_sh, "kernel_K": K,
        "mean_soup_gt_metrics": {k: mean[k] for k in ("l1", "psnr", "ssim")},
        "render_ms_per_frame_with_readback": mean["ms"],
        "fps_with_readback": 1000.0 / mean["ms"] if mean["ms"] > 0 else None,
        "bench_ms_per_frame": bench_ms,
        "bench_fps": (1000.0 / bench_ms) if bench_ms else None,
        "profile_ms": ({k: prof[k] / max(prof["n"], 1) * 1000 for k in ("sort", "color", "upload", "draw", "readback")}
                       if prof["n"] else None),
        # viewer FPS = what a real-time display achieves (no CPU readback; SH/sort are on-GPU work a viewer also does)
        "viewer_ms_per_frame": (sum(prof[k] for k in ("sort", "upload", "draw")) / max(prof["n"], 1) * 1000
                                if prof["n"] else None),
        "viewer_fps": (1000.0 / (sum(prof[k] for k in ("sort", "upload", "draw")) / max(prof["n"], 1) * 1000)
                       if prof["n"] else None),
        "per_view": per_view,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "per_view"}, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        json.dump(result, open(args.output, "w"), indent=2)


if __name__ == "__main__":
    main()
