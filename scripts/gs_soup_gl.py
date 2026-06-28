"""3DGS billboard 'soup' renderer — the SAME hardware-rasterization pipeline as scripts/drk_soup_gl.py
(instanced quads + fragment falloff + per-primitive global depth sort + single-pass alpha blend +
RGBA16F), but each primitive is a 3D Gaussian rendered as an EWA splat billboard. Used for a fair
DRK-vs-3DGS comparison at equal rendering method: reports soup-vs-GT PSNR/SSIM and viewer FPS.

EWA math ported from submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/forward.cu
(computeCov2D Jacobian + 0.3 low-pass, conic = inv(cov2D), 3-sigma radius, power/alpha).
cov3D (Sigma, 6 floats) is precomputed in Python via GaussianModel.get_covariance to avoid the
glm column-major quat->matrix ambiguity.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from argparse import Namespace
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.image_utils import psnr as psnr_fn
from utils.loss_utils import ssim as ssim_fn
from utils.sh_utils import eval_sh
import moderngl

# STRIDE=58: [0:3]mean [3]opacity [4:10]cov3D(Sigma 6) [10:58]SH(16*3)
VERT_SRC = """
#version 430
layout(std430, binding=0) buffer Static { float sdata[]; };
layout(std430, binding=2) buffer Order  { int order[]; };
uniform mat4 u_mvp;      // full_proj_transform (raw, column-major)
uniform mat4 u_view;     // world_view_transform (raw)
uniform vec3 u_campos; uniform vec2 u_res; uniform int u_sh_degree; uniform int u_stride;
uniform float u_focal_x, u_focal_y, u_tan_fovx, u_tan_fovy;
out vec2 v_d0; flat out vec3 v_conic; flat out vec2 v_center; flat out float v_opacity; flat out vec3 v_rgb;
float gf(int p,int i){return sdata[p*u_stride+i];}
vec3 shc(int p,int i){int b=10+i*3;return vec3(gf(p,b),gf(p,b+1),gf(p,b+2));}
vec3 evalSH(int p,vec3 d){float x=d.x,y=d.y,z=d.z,xx=x*x,yy=y*y,zz=z*z,xy=x*y,yz=y*z,xz=x*z;
 vec3 r=0.28209479177387814*shc(p,0);
 if(u_sh_degree>0)r+=-0.4886025119029199*y*shc(p,1)+0.4886025119029199*z*shc(p,2)-0.4886025119029199*x*shc(p,3);
 if(u_sh_degree>1)r+=1.0925484305920792*xy*shc(p,4)-1.0925484305920792*yz*shc(p,5)+0.31539156525252005*(2.0*zz-xx-yy)*shc(p,6)-1.0925484305920792*xz*shc(p,7)+0.5462742152960396*(xx-yy)*shc(p,8);
 if(u_sh_degree>2)r+=-0.5900435899266435*y*(3.0*xx-yy)*shc(p,9)+2.890611442640554*xy*z*shc(p,10)-0.4570457994644658*y*(4.0*zz-xx-yy)*shc(p,11)+0.3731763325901154*z*(2.0*zz-3.0*xx-3.0*yy)*shc(p,12)-0.4570457994644658*x*(4.0*zz-xx-yy)*shc(p,13)+1.445305721320277*z*(xx-yy)*shc(p,14)-0.5900435899266435*x*(xx-3.0*yy)*shc(p,15);
 return max(r+vec3(0.5),vec3(0.0));}
void main(){
 int p=order[gl_InstanceID];
 vec3 mean=vec3(gf(p,0),gf(p,1),gf(p,2));
 float opacity=gf(p,3);
 mat3 Sig=mat3(gf(p,4),gf(p,5),gf(p,6), gf(p,5),gf(p,7),gf(p,8), gf(p,6),gf(p,8),gf(p,9));
 // view-space mean
 vec3 t=(u_view*vec4(mean,1.0)).xyz;
 float lx=1.3*u_tan_fovx, ly=1.3*u_tan_fovy;
 t.x=min(lx,max(-lx,t.x/t.z))*t.z; t.y=min(ly,max(-ly,t.y/t.z))*t.z;
 mat3 J=mat3(u_focal_x/t.z,0.0,-(u_focal_x*t.x)/(t.z*t.z), 0.0,u_focal_y/t.z,-(u_focal_y*t.y)/(t.z*t.z), 0.0,0.0,0.0);
 mat3 W=transpose(mat3(u_view));  // CUDA glm W is built from viewmatrix[0,4,8,...] = transpose of mat3(u_view)
 mat3 T=W*J;
 mat3 cov=transpose(T)*Sig*T;
 float A=cov[0][0]+0.3, B=cov[0][1], C=cov[1][1]+0.3;
 float det=A*C-B*B;
 vec4 ph=u_mvp*vec4(mean,1.0);
 if(det==0.0 || ph.w<=0.0){gl_Position=vec4(2.0,2.0,2.0,1.0);return;}  // cull
 vec3 conic=vec3(C/det,-B/det,A/det);
 vec3 pp=ph.xyz/ph.w;
 vec2 center=vec2(((pp.x+1.0)*u_res.x-1.0)*0.5, ((pp.y+1.0)*u_res.y-1.0)*0.5);  // pixel (CV, y-down)
 float mid=0.5*(A+C); float disc=sqrt(max(0.1,mid*mid-det));
 float radius=ceil(3.0*sqrt(max(mid+disc,mid-disc)));
 int vid=gl_VertexID; vec2 corner = (vid==0)?vec2(-1,-1):(vid==1)?vec2(1,-1):(vid==2)?vec2(1,1):(vid==3)?vec2(-1,-1):(vid==4)?vec2(1,1):vec2(-1,1);
 vec2 ndc=pp.xy + corner*vec2(2.0*radius/u_res.x, 2.0*radius/u_res.y);
 gl_Position=vec4(ndc, pp.z, 1.0);
 v_conic=conic; v_center=center; v_opacity=opacity;
 v_rgb=evalSH(p, normalize(mean-u_campos));
}
"""
FRAG_SRC = """
#version 430
flat in vec3 v_conic; flat in vec2 v_center; flat in float v_opacity; flat in vec3 v_rgb; in vec2 v_d0;
uniform vec2 u_res;
out vec4 frag;
void main(){
 // CUDA ndc2Pix and the GL viewport share the same origin (0 at ndc=-1), so no y-flip here
 vec2 d=v_center-gl_FragCoord.xy;
 float power=-0.5*(v_conic.x*d.x*d.x + v_conic.z*d.y*d.y) - v_conic.y*d.x*d.y;
 if(power>0.0) discard;
 float a=min(0.99, v_opacity*exp(power));
 if(a<1.0/255.0) discard;
 frag=vec4(max(v_rgb,vec3(0.0)), a);
}
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--source-path", required=True)
    p.add_argument("-m", "--model-path", required=True)
    p.add_argument("--load-iteration", type=int, default=-1)
    p.add_argument("--split", choices=["train", "test", "all"], default="test")
    p.add_argument("--views", type=int, default=-1)
    p.add_argument("--supersample", type=int, default=1)
    p.add_argument("--gl-device", type=int, default=0)
    p.add_argument("--render-sh-degree", type=int, default=-1)
    p.add_argument("--keep", type=int, default=-1, help="keep top-N gaussians by opacity (for prim-count curve); -1=all")
    p.add_argument("--bench-iters", type=int, default=0)
    p.add_argument("--save-images", default=None)
    p.add_argument("--output", default=None)
    return p.parse_args()


def make_dataset_args(a):
    return Namespace(source_path=os.path.abspath(a.source_path), model_path=os.path.abspath(a.model_path),
                     images="images", resolution=-1, white_background=False, data_device="cuda", eval=True,
                     val_as_train=False, gs_type="GS", metric_masked=False)


@torch.no_grad()
def build_gs(model, keep):
    mean = model.get_xyz
    opacity = model.get_opacity.reshape(-1)
    cov3d = model.get_covariance(1.0)            # [N,6] symmetric Sigma (s00,s01,s02,s11,s12,s22)
    feats = model.get_features                   # [N,16,3]
    N = mean.shape[0]
    if keep and keep > 0 and keep < N:
        idx = torch.topk(opacity, keep).indices
        mean, opacity, cov3d, feats = mean[idx], opacity[idx], cov3d[idx], feats[idx]
        N = keep
    sh = feats.reshape(N, -1)                     # [N,48]
    STRIDE = 58
    s = torch.zeros((N, STRIDE), device=mean.device)
    s[:, 0:3] = mean; s[:, 3] = opacity; s[:, 4:10] = cov3d; s[:, 10:58] = sh
    return s.contiguous().cpu().numpy().astype("f4"), STRIDE, N, mean.detach()


def img_metrics(pred, tgt):
    return {"l1": torch.abs(pred - tgt).mean().item(),
            "psnr": psnr_fn(pred[None], tgt[None]).mean().item(),
            "ssim": ssim_fn(pred, tgt).item()}


def main():
    args = parse_args()
    ckpt_dir = os.path.join(os.path.abspath(args.model_path), "point_cloud")
    model = GaussianModel(3)
    scene = Scene(make_dataset_args(args), model, load_iteration=args.load_iteration, shuffle=False)
    render_sh = min(3, 4) if args.render_sh_degree < 0 else min(args.render_sh_degree, 3)
    model.eval() if hasattr(model, "eval") else None

    cams = (scene.getTestCameras() if args.split == "test" else
            scene.getTrainCameras() if args.split == "train" else
            scene.getTrainCameras() + scene.getTestCameras())
    if len(cams) == 0:
        cams = scene.getTrainCameras()
    if args.views > 0:
        cams = cams[:args.views]
    W = int(cams[0].image_width); H = int(cams[0].image_height); ss = args.supersample
    rW, rH = W * ss, H * ss

    ctx = moderngl.create_context(standalone=True, backend="egl", require=430, device_index=int(args.gl_device))
    prog = ctx.program(vertex_shader=VERT_SRC, fragment_shader=FRAG_SRC)
    sdata, STRIDE, N, means_t = build_gs(model, args.keep)
    prog["u_stride"] = STRIDE; prog["u_res"] = (float(rW), float(rH)); prog["u_sh_degree"] = int(render_sh)
    static_buf = ctx.buffer(sdata.tobytes())
    order_buf = ctx.buffer(reserve=N * 4, dynamic=True)
    vao = ctx.vertex_array(prog, [])
    fbo = ctx.framebuffer(color_attachments=[ctx.texture((rW, rH), 4, dtype="f2")])

    ones_t = torch.ones((N, 1), device=means_t.device)
    prof = {"sort": 0.0, "upload": 0.0, "draw": 0.0, "readback": 0.0, "n": 0}

    @torch.no_grad()
    def render(cam, record=False):
        def tick(): torch.cuda.synchronize(); return time.time()
        t = tick()
        wv = cam.world_view_transform; fp = cam.full_proj_transform
        homo = torch.cat([means_t, ones_t], dim=1)
        clip = homo @ fp; w = clip[:, 3]; ndc = clip[:, :3] / w.unsqueeze(1).clamp_min(1e-6)
        vis = (w > 0.01) & (ndc[:, 0].abs() < 1.3) & (ndc[:, 1].abs() < 1.3) & (ndc[:, 2] > -1.0) & (ndc[:, 2] < 1.0)
        depth_v = (homo @ wv)[:, 2]
        vis_idx = torch.nonzero(vis, as_tuple=False).squeeze(1)
        order = vis_idx[torch.argsort(depth_v[vis_idx], descending=True)].to(torch.int32).cpu().numpy()
        n_inst = int(order.shape[0]); t1 = tick()
        order_buf.write(order.tobytes())
        static_buf.bind_to_storage_buffer(0); order_buf.bind_to_storage_buffer(2)
        prog["u_mvp"].write(np.ascontiguousarray(fp.detach().cpu().numpy(), "f4").tobytes())
        prog["u_view"].write(np.ascontiguousarray(wv.detach().cpu().numpy(), "f4").tobytes())
        prog["u_campos"].write(cam.camera_center.detach().cpu().numpy().astype("f4").tobytes())
        import math as _m
        prog["u_tan_fovx"] = _m.tan(cam.FoVx * 0.5); prog["u_tan_fovy"] = _m.tan(cam.FoVy * 0.5)
        prog["u_focal_x"] = rW / (2 * _m.tan(cam.FoVx * 0.5)); prog["u_focal_y"] = rH / (2 * _m.tan(cam.FoVy * 0.5))
        fbo.use(); ctx.viewport = (0, 0, rW, rH); ctx.clear(0.0, 0.0, 0.0, 0.0); ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND); ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        t2 = time.time()
        vao.render(mode=moderngl.TRIANGLES, vertices=6, instances=n_inst); ctx.finish()
        t3 = time.time()
        raw = np.frombuffer(fbo.read(components=4, dtype="f2"), dtype="float16").reshape(rH, rW, 4)
        img = torch.from_numpy(np.ascontiguousarray(raw[..., :3])).to("cuda").permute(2, 0, 1).contiguous().float()
        if ss > 1: img = torch.nn.functional.avg_pool2d(img[None], ss)[0]
        t4 = tick()
        if record:
            prof["sort"] += t1 - t; prof["upload"] += t2 - t1; prof["draw"] += t3 - t2; prof["readback"] += t4 - t3; prof["n"] += 1
        return img.clamp(0, 1)

    per_view = []
    if args.save_images: os.makedirs(args.save_images, exist_ok=True)
    _ = render(cams[0])
    for i, cam in enumerate(cams):
        img = render(cam)
        gt = cam.original_image.to("cuda").clamp(0, 1)
        per_view.append(img_metrics(img, gt))
        if args.save_images:
            from PIL import Image
            Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")).save(os.path.join(args.save_images, f"gs_{i:03d}.png"))
    if args.bench_iters > 0:
        for _ in range(5): render(cams[0])
        for _ in range(args.bench_iters): render(cams[0], record=True)
    mean = {k: float(np.mean([v[k] for v in per_view])) for k in ("l1", "psnr", "ssim")}
    nb = max(prof["n"], 1)
    pm = {k: prof[k] / nb * 1000 for k in ("sort", "upload", "draw", "readback")}
    viewer_ms = pm["sort"] + pm["upload"] + pm["draw"]
    result = {"model_path": os.path.abspath(args.model_path), "load_iteration": scene.loaded_iter,
              "primitives": int(N), "resolution": [W, H], "supersample": ss, "render_sh_degree": render_sh,
              "mean_soup_gt_metrics": mean, "profile_ms": pm if prof["n"] else None,
              "viewer_ms_per_frame": viewer_ms if prof["n"] else None,
              "viewer_fps": (1000.0 / viewer_ms) if prof["n"] and viewer_ms > 0 else None, "per_view": per_view}
    print(json.dumps({k: v for k, v in result.items() if k != "per_view"}, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        json.dump(result, open(args.output, "w"), indent=2)


if __name__ == "__main__":
    main()
