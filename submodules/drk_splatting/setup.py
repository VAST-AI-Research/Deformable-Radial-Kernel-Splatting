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

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import shutil
os.path.dirname(os.path.abspath(__file__))

kernel_k = int(os.environ.get("DRK_KERNEL_K", "8"))
if kernel_k < 3 or kernel_k > 16:
    raise ValueError("DRK_KERNEL_K must be in [3, 16]")


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


low_pass_filter = env_bool("DRK_LOW_PASS_FILTER", True)
low_pass_define = "1" if low_pass_filter else "0"
force_l1_kernel = env_bool("DRK_FORCE_L1_KERNEL", False)
force_l1_define = "1" if force_l1_kernel else "0"
force_acutance_one_kernel = env_bool("DRK_FORCE_ACUTANCE_ONE_KERNEL", False)
force_acutance_one_define = "1" if force_acutance_one_kernel else "0"
screen_space_l1_kernel = env_bool("DRK_SCREEN_SPACE_L1_KERNEL", False)
screen_space_l1_define = "1" if screen_space_l1_kernel else "0"
tight_exact_aabb = env_bool("DRK_TIGHT_EXACT_AABB", False)
tight_exact_aabb_define = "1" if tight_exact_aabb else "0"
tight_exact_aabb_margin = float(os.environ.get("DRK_TIGHT_EXACT_AABB_MARGIN", "2.0"))
tight_general_aabb = env_bool("DRK_TIGHT_GENERAL_AABB", True)
tight_general_aabb_define = "1" if tight_general_aabb else "0"
max_layers = int(os.environ.get("DRK_MAX_LAYERS", "0"))
pixel_center_offset_x = float(os.environ.get("DRK_PIXEL_CENTER_OFFSET_X", "0.0"))
pixel_center_offset_y = float(os.environ.get("DRK_PIXEL_CENTER_OFFSET_Y", "0.0"))
pixel_center_offset_apply_projection = env_bool("DRK_PIXEL_CENTER_OFFSET_APPLY_PROJECTION", True)
pixel_center_offset_apply_ray = env_bool("DRK_PIXEL_CENTER_OFFSET_APPLY_RAY", True)
pixel_center_offset_apply_projection_define = "1" if pixel_center_offset_apply_projection else "0"
pixel_center_offset_apply_ray_define = "1" if pixel_center_offset_apply_ray else "0"


class DRKBuildExtension(BuildExtension):
    def build_extensions(self):
        stamp = os.path.join(self.build_temp, "drk_build_config.txt")
        build_config = (
            f"KERNEL_K={kernel_k}\n"
            f"LOW_PASS_FILTER={low_pass_define}\n"
            f"DRK_FORCE_L1_KERNEL={force_l1_define}\n"
            f"DRK_FORCE_ACUTANCE_ONE_KERNEL={force_acutance_one_define}\n"
            f"DRK_SCREEN_SPACE_L1_KERNEL={screen_space_l1_define}\n"
            f"DRK_TIGHT_EXACT_AABB={tight_exact_aabb_define}\n"
            f"DRK_TIGHT_EXACT_AABB_MARGIN={tight_exact_aabb_margin}\n"
            f"DRK_TIGHT_GENERAL_AABB={tight_general_aabb_define}\n"
            f"DRK_MAX_LAYERS={max_layers}\n"
            f"DRK_PIXEL_CENTER_OFFSET_X={pixel_center_offset_x}\n"
            f"DRK_PIXEL_CENTER_OFFSET_Y={pixel_center_offset_y}\n"
            f"DRK_PIXEL_CENTER_OFFSET_APPLY_PROJECTION={pixel_center_offset_apply_projection_define}\n"
            f"DRK_PIXEL_CENTER_OFFSET_APPLY_RAY={pixel_center_offset_apply_ray_define}\n"
        )
        old_config = None
        if os.path.exists(stamp):
            with open(stamp) as f:
                old_config = f.read()
        if old_config != build_config and os.path.isdir(self.build_temp):
            shutil.rmtree(self.build_temp)
        os.makedirs(self.build_temp, exist_ok=True)
        super().build_extensions()
        with open(stamp, "w") as f:
            f.write(build_config)

setup(
    name="drk_splatting",
    packages=['drk_splatting'],
    ext_modules=[
        CUDAExtension(
            name="drk_splatting._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={
                "cxx": [
                    f"-DKERNEL_K={kernel_k}",
                    f"-DLOW_PASS_FILTER={low_pass_define}",
                    f"-DDRK_FORCE_L1_KERNEL={force_l1_define}",
                    f"-DDRK_FORCE_ACUTANCE_ONE_KERNEL={force_acutance_one_define}",
                    f"-DDRK_SCREEN_SPACE_L1_KERNEL={screen_space_l1_define}",
                    f"-DDRK_TIGHT_EXACT_AABB={tight_exact_aabb_define}",
                    f"-DDRK_TIGHT_EXACT_AABB_MARGIN={tight_exact_aabb_margin}",
                    f"-DDRK_TIGHT_GENERAL_AABB={tight_general_aabb_define}",
                    f"-DDRK_MAX_LAYERS={max_layers}",
                    f"-DDRK_PIXEL_CENTER_OFFSET_X={pixel_center_offset_x}",
                    f"-DDRK_PIXEL_CENTER_OFFSET_Y={pixel_center_offset_y}",
                    f"-DDRK_PIXEL_CENTER_OFFSET_APPLY_PROJECTION={pixel_center_offset_apply_projection_define}",
                    f"-DDRK_PIXEL_CENTER_OFFSET_APPLY_RAY={pixel_center_offset_apply_ray_define}",
                ],
                "nvcc": [
                    "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
                    "--use_fast_math",
                    "-O3",
                    "--ftz=true",  # Flush denormals to zero
                    "-lineinfo",  # Keep line info for profiling
                    f"-DKERNEL_K={kernel_k}",
                    f"-DLOW_PASS_FILTER={low_pass_define}",
                    f"-DDRK_FORCE_L1_KERNEL={force_l1_define}",
                    f"-DDRK_FORCE_ACUTANCE_ONE_KERNEL={force_acutance_one_define}",
                    f"-DDRK_SCREEN_SPACE_L1_KERNEL={screen_space_l1_define}",
                    f"-DDRK_TIGHT_EXACT_AABB={tight_exact_aabb_define}",
                    f"-DDRK_TIGHT_EXACT_AABB_MARGIN={tight_exact_aabb_margin}",
                    f"-DDRK_TIGHT_GENERAL_AABB={tight_general_aabb_define}",
                    f"-DDRK_MAX_LAYERS={max_layers}",
                    f"-DDRK_PIXEL_CENTER_OFFSET_X={pixel_center_offset_x}",
                    f"-DDRK_PIXEL_CENTER_OFFSET_Y={pixel_center_offset_y}",
                    f"-DDRK_PIXEL_CENTER_OFFSET_APPLY_PROJECTION={pixel_center_offset_apply_projection_define}",
                    f"-DDRK_PIXEL_CENTER_OFFSET_APPLY_RAY={pixel_center_offset_apply_ray_define}",
                ]
            })
        ],
    cmdclass={
        'build_ext': DRKBuildExtension
    }
)
