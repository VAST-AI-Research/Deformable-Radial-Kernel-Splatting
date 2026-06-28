<h1 align="center">
  Deformable Radial Kernel Splatting

  <a href="https://www.hku.hk/"><img height="70" src="assets/HKU.png"> </a>
  <a href="https://github.com/VAST-AI-Research/"><img height="70" src="assets/VAST.png"> </a>
</h1>

This repository contains the code for **Deformable Radial Kernel Splatting (DRK)**.

<div align="center">
  
[![Website](assets/badge-website.svg)](https://yihua7.github.io/DRK-web/)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/pdf/2412.11752)

</div>

**Deformable Radial Kernel (DRK)** extends Gaussian kernels with learnable radial bases, enabling the modeling of diverse shape primitives. It introduces parameters to control the sharpness and boundary curvature of these primitives. The following video showcases the effectiveness of each parameter:

<div align="center">
  <img src="assets/DRK-Parameters.gif" width="100%">
</div>

DRK can flexibly fit various basic primitives with diverse shapes and sharp boundaries:

<div align="center">
  <img src="assets/Shape-Fitting.gif" width="100%">
</div>

---

## Updated densification strategy

DRK with an updated densification strategy, compared to 3DGS on **Tanks & Temples** (Family, COLMAP poses, 896×512, test split):

| Family | PSNR | SSIM | L1 | prims |
|---|---|---|---|---|
| 3DGS | 22.25 | 0.782 | 0.047 | 1.70M |
| DRK | 22.46 | 0.765 | 0.045 | 0.90M |

---

## Real-time rendering via the traditional pipeline (web / mobile)

A trained **soft-DRK** model can be rendered as a **triangle soup** by a standard hardware
rasterization pipeline — each primitive becomes a billboard whose **fragment shader** evaluates the
DRK radial kernel, composited with per-primitive depth sort + single-pass alpha blend. This runs in
**WebGL2 / OpenGL ES** (web, mobile, headsets) with no CUDA dependency, while preserving the soft DRK
appearance. `scripts/drk_soup_gl.py` is the reference renderer; `scripts/web/viewer.html` is a
self-contained WebGL2 viewer.

### Web / mobile deployment

```bash
# export a trained DRK model to a portable WebGL viewer folder (geometry + attributes + GLSL)
python scripts/export_soup_web.py -m <run_dir> --load-iteration <N> --out viewer_out
# view it — WebGL2 == the mobile GLES path, so in-browser FPS gauges on-device feasibility
cd viewer_out && python3 -m http.server 8000   # open http://localhost:8000
```

Add `--embed` for a single double-click HTML, or `--skybox <asset_dir>` to composite a learned
multi-shell background panorama behind the foreground. A ready-to-open example (50k-primitive
Family soup + skybox, no server needed) is in [`demo/family_50k_viewer.html`](demo/family_50k_viewer.html)
— open it in any browser (drag = orbit, scroll = zoom, slider = visible-primitive radius / FPS).

### Comparison with 3DGS (same WebGL pipeline)

Both methods rendered as billboard soups in the **same** pipeline (Tanks & Temples *Family*, test
split, 896×512; FPS excludes offline read-back):

| method | primitives | PSNR | FPS |
|---|---|---|---|
| 3DGS | 1.70 M | 21.5 | 116 |
| **DRK (ours)** | **50 k** | 20.8 | **186** |
| **DRK (ours)** | 150 k | 21.0 | 81 |

DRK reaches comparable quality with **~11–34× fewer primitives** (3DGS collapses if subsampled),
yielding a **~19× smaller asset** (≈21 MB vs ≈395 MB) and higher FPS at the low primitive counts
mobile devices need. Reproduce with `scripts/gs_soup_gl.py` + `scripts/export_compare_web.py`
(`scripts/web/compare.html` toggles DRK ⇄ 3DGS live).

---

## Environment Setup

### Create and Activate Python Environment
#### Using Conda:
```bash
conda create -n drkenv python=3.9  # (Python >= 3.8)
conda activate drkenv
```

#### Using Virtualenv:
```bash
virtualenv drkenv -p python3.9  # (Python >= 3.8)
source drkenv/bin/activate
```

### Install Dependencies
```bash
python -m pip install -U pip setuptools wheel importlib-metadata
python -m pip install -r requirements.txt

# Use the CUDA toolkit that matches the PyTorch wheels above.
source ./switch-cuda.sh 11.8

cd submodules/depth-diff-gaussian-rasterization
python -m pip install --no-build-isolation --no-deps .
cd ../drk_splatting
python -m pip install --no-build-isolation --no-deps .
cd ../simple-knn
python -m pip install --no-build-isolation --no-deps .
cd ../..
```

---

## UI Demo

We provide a UI demo to better understand the effects of DRK attributes and cache-sorting. To run the demo, execute the following script:

```bash
python drk_demo.py
```

The demo allows you to adjust attribute bars, switch rendering modes (normal, alpha, depth, RGB), toggle cache-sorting, and explore DRK's flexible representation capabilities.

<div align="center">
  <img src="assets/drk_demo.gif" width="100%">
</div>

---

## Mesh2DRK

We also provide a script to convert mesh assets into DRK representation **without training**. To achieve mixed rendering of meshes and reconstructed scenes, specify the `scene_path` in [mesh2drk.py](./mesh2drk.py). If `scene_path` is left empty, the script will render the mesh only. You can modify the `mesh_path_list` to include any assets you wish to render. Currently, `.obj + .mtl` and `.ply` formats are supported. For reference, we provide example assets in the [meshes](./meshes) folder.

```bash
python mesh2drk.py
```

<div align="center">
  <img src="assets/mixed_rendering.gif" width="100%">
</div>

---

## Data Download

Download the datasets using the following links:

- [MipNeRF-360](https://jonbarron.info/mipnerf360)
- [DiverseScenes](https://drive.google.com/file/d/1k1Eb_0K6Bo3VS33cpwOmqHLdlQJrGyQy/view?usp=sharing)

---

## Running the Code

### Commands
Run the following commands in your terminal:

#### Training:
```bash
CUDA_VISIBLE_DEVICES=${GPU} python train.py -s ${PATH_TO_DATA} -m ${LOG_PATH} --eval --gs_type DRK --kernel_density dense --cache_sort  # Optional: --gui --is_unbounded
```

#### Evaluation:
```bash
CUDA_VISIBLE_DEVICES=${GPU} python train.py -s ${PATH_TO_DATA} -m ${LOG_PATH} --eval --gs_type DRK --kernel_density dense --cache_sort --metric
```

### Command Options:
- `--kernel_density`: Specifies the primitive density (number) for reconstruction. Choose from `dense`, `middle`, or `sparse`.
- `--cache_sort`: (Optional) Use cache sorting to avoid popping artifacts and slightly increase PSNR (approx. +0.1dB). Ensure consistency between training and evaluation. Note: In specular scenes, disabling cache-sort may yield better results as highlights are better modeled without strict sorting.
- `--is_unbounded`: Use different hyperparameters for unbounded scenes (e.g., Mip360).
- `--gui`: Enables an interactive visualization UI. Toggle cache-sorting, tile-culling, and view different rendering modes (normal, depth, alpha) via the control panel.

### Batch Scripts
Scripts for evaluating all scenes in the dataset are provided in the [scripts](./scripts) folder. Modify the paths in the scripts before running them.

```bash
python ./scripts/diverse_script.py  # For DiverseScenes
python ./scripts/mip360_script.py   # For MipNeRF-360
```

---

## Recent Optimizations

### CUDA Optimization
- **Precomputed kernel vectors**: `scale * [cos(θ), sin(θ)]` computed once in preprocess, reused in rendering and tile culling
- **`atan2f` replaces `acos+sqrt`**: faster angle computation in inner loop
- **Removed `roundf` truncation**: eliminated expensive per-hit rounding
- **Shared memory optimization**: geometry buffer strategy to stay within 48KB limit
- **Densification stats via CUDA**: collect absolute gradients (`fabsf`) directly in backward kernel, avoiding Python overhead
- **Branchless segment search**: replaced branch-heavy linear scan with predicated additions for better warp coherence
- **Fast math intrinsics**: `__expf`, `__cosf`, `__sincosf`, `__frcp_rn` to replace standard `exp`/`cos`/`sin`/division
- **Cached reciprocals**: pre-compute `1/delta`, `1/(scale²)`, `1/(theta_r−theta_l)`, `1/dir_dot_n` etc. to eliminate redundant divisions
- **Compiler flags**: `--use_fast_math -O3 --ftz=true` in `setup.py`

### Rendering Quality
- **Opacity-gradient driven densification**: combine position and opacity gradients for more accurate densification decisions
- **Visibility-aware pruning**: prune low-visibility + low-opacity floaters during densification
- **Multi-scale anti-aliasing loss** (`--lambda_multiscale`): optional multi-resolution L1+SSIM supervision
- **Opacity regularization** (`--lambda_opacity_reg`): entropy-based regularization to suppress semi-transparent floaters

### Misc
- Fixed install instructions and added `.gitignore` for build outputs
- `simple_knn.cu`: use `<float.h>` instead of hardcoded `FLT_MAX`
- `gui_utils`: graceful fallback when `dearpygui` is not installed

---

## Citing

If you find our work useful, please consider citing:

```bibtex
@article{huang2024deformable,
  title={Deformable Radial Kernel Splatting},
  author={Huang, Yi-Hua and Lin, Ming-Xian and Sun, Yang-Tian and Yang, Ziyi and Lyu, Xiaoyang and Cao, Yan-Pei and Qi, Xiaojuan},
  journal={arXiv preprint arXiv:2412.11752},
  year={2024}
}
```
