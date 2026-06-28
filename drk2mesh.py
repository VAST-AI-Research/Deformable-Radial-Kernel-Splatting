import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import numpy as np
from plyfile import PlyData, PlyElement


C0 = 0.28209479177387814
DEFAULT_ACUTANCE_MIN = -0.1
DEFAULT_ACUTANCE_MAX = 0.99
DEFAULT_SUPPORT_ALPHA = math.exp(-4.5)


def support_alpha_to_scale_modifier(support_alpha: float) -> float:
    if not (0.0 < float(support_alpha) < 1.0):
        raise ValueError("--support-alpha must be in (0, 1)")
    return math.sqrt(-2.0 * math.log(float(support_alpha))) / 3.0


def scale_modifier_to_support_alpha(scale_modifier: float) -> float:
    if float(scale_modifier) <= 0.0:
        raise ValueError("--scale-modifier must be positive")
    return math.exp(-0.5 * (3.0 * float(scale_modifier)) ** 2)


def resolve_scale_modifier(scale_modifier: float, support_alpha: Optional[float]) -> float:
    if support_alpha is None:
        if float(scale_modifier) <= 0.0:
            raise ValueError("--scale-modifier must be positive")
        return float(scale_modifier)
    if abs(float(scale_modifier) - 1.0) > 1e-9:
        raise ValueError("--support-alpha and non-default --scale-modifier are mutually exclusive")
    return support_alpha_to_scale_modifier(float(support_alpha))


def sigmoid(x):
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def sorted_fields(names: Iterable[str], prefix: str):
    return sorted(
        [name for name in names if name.startswith(prefix)],
        key=lambda name: int(name.split("_")[-1]),
    )


def stack_fields(vertex, names):
    if len(names) == 0:
        return np.empty((len(vertex), 0), dtype=np.float32)
    return np.stack([np.asarray(vertex[name], dtype=np.float32) for name in names], axis=1)


def embedded_sh_coeff_count(names):
    names = set(names)
    coeff_count = 0
    while all(f"drk_sh_{coeff_count}_{channel}" in names for channel in ("r", "g", "b")):
        coeff_count += 1
    return coeff_count


def optional_ply_element(ply_data, name):
    try:
        return ply_data[name]
    except KeyError:
        return None


def parse_drk_ply_metadata(ply_data):
    metadata = {}
    for comment in getattr(ply_data, "comments", []) or []:
        parts = str(comment).strip().split(None, 1)
        if len(parts) == 2 and parts[0].startswith("drk_"):
            metadata[parts[0]] = parts[1]
    return metadata


def metadata_float(metadata, key):
    value = metadata.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def metadata_int(metadata, key):
    value = metadata.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def metadata_bool(metadata, key):
    value = metadata.get(key)
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    return None


def resolve_acutance_range(ply_data, acutance_min=None, acutance_max=None):
    if acutance_min is not None and acutance_max is not None:
        return float(acutance_min), float(acutance_max), "args"
    if acutance_min is not None or acutance_max is not None:
        raise ValueError("Set both acutance_min and acutance_max, or leave both unset for metadata/default")

    metadata = parse_drk_ply_metadata(ply_data)
    meta_min = metadata_float(metadata, "drk_acutance_min")
    meta_max = metadata_float(metadata, "drk_acutance_max")
    if meta_min is not None and meta_max is not None:
        if meta_min >= meta_max:
            raise ValueError(f"Invalid DRK acutance metadata range: {meta_min} >= {meta_max}")
        return meta_min, meta_max, "ply_comment"

    return DEFAULT_ACUTANCE_MIN, DEFAULT_ACUTANCE_MAX, "default"


def load_drk_metadata(path, acutance_min=None, acutance_max=None):
    ply_data = PlyData.read(path)
    resolved_min, resolved_max, source = resolve_acutance_range(ply_data, acutance_min, acutance_max)
    return {
        "acutance_min": float(resolved_min),
        "acutance_max": float(resolved_max),
        "acutance_range_source": source,
        "sh_degree": int(infer_sh_degree_from_property_names(ply_data["vertex"].data.dtype.names)),
        "ply_comments": parse_drk_ply_metadata(ply_data),
    }


def infer_sh_degree_from_property_names(names):
    rest_count = len([name for name in names if name.startswith("f_rest_")])
    basis_count = (rest_count + 3) // 3
    degree = int(round(math.sqrt(float(basis_count)) - 1.0))
    if rest_count != 3 * (degree + 1) ** 2 - 3:
        raise ValueError(f"Cannot infer SH degree from {rest_count} f_rest_* fields")
    return degree


def infer_sh_degree_from_ply(path):
    ply_data = PlyData.read(path)
    return infer_sh_degree_from_property_names(ply_data["vertex"].data.dtype.names)


def quaternion_to_matrix(quaternions):
    q = quaternions.astype(np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    r, i, j, k = [q[:, idx] for idx in range(4)]
    two_s = 2.0 / np.maximum(np.sum(q * q, axis=-1), 1e-8)
    mats = np.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        axis=1,
    )
    return mats.reshape((-1, 3, 3))


@dataclass
class ExportStats:
    input_path: str
    output_path: str
    input_primitives: int
    exported_primitives: int
    kernel_k: int
    vertices: int
    triangles: int
    min_opacity: float
    max_scale: Optional[float]
    max_scale_quantile: Optional[float]
    max_scale_threshold: Optional[float]
    max_scale_area: Optional[float]
    max_scale_area_quantile: Optional[float]
    max_scale_area_threshold: Optional[float]
    mean_opacity: float
    mean_acutance: float
    mean_l1l2_rate: float
    acutance_min: float
    acutance_max: float
    acutance_range_source: str
    topk_score: str
    boundary_alpha: float
    scale_modifier: float
    support_alpha: Optional[float]
    boundary_scale: float
    boundary_scale_min: float
    boundary_scale_max: float
    boundary_opacity_power: float
    boundary_opacity_reference: float
    force_acutance: Optional[float]
    force_l1l2_rate: Optional[float]
    force_opacity: Optional[float]
    rings: int
    angular_subdivisions: int
    has_vertex_alpha: bool


def load_drk(path, acutance_min=None, acutance_max=None, theta_residual=0.0, return_metadata=False, force_acutance=None, force_l1l2_rate=None, force_opacity=None):
    ply_data = PlyData.read(path)
    acutance_min, acutance_max, acutance_range_source = resolve_acutance_range(
        ply_data,
        acutance_min,
        acutance_max,
    )
    vertex = ply_data["vertex"].data
    names = vertex.dtype.names

    required = ["x", "y", "z", "opacity", "acutance", "f_dc_0", "f_dc_1", "f_dc_2"]
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"{path} is missing DRK fields: {', '.join(missing)}")

    scale_names = sorted_fields(names, "scale_")
    theta_names = sorted_fields(names, "theta_")
    l1l2_names = sorted_fields(names, "l1l2rate_")
    rot_names = sorted_fields(names, "rot")
    if len(scale_names) == 0 or len(scale_names) != len(theta_names):
        raise ValueError("DRK PLY must contain matching scale_* and theta_* fields")
    if len(l1l2_names) != 1:
        raise ValueError("DRK PLY must contain one l1l2rate_* field")
    if len(rot_names) != 4:
        raise ValueError("DRK PLY must contain quaternion fields rot_0..rot_3")

    xyz = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )
    colors_dc = np.stack(
        [
            np.asarray(vertex["f_dc_0"], dtype=np.float32),
            np.asarray(vertex["f_dc_1"], dtype=np.float32),
            np.asarray(vertex["f_dc_2"], dtype=np.float32),
        ],
        axis=1,
    )
    colors = np.clip(colors_dc * C0 + 0.5, 0.0, 1.0)

    scales = np.exp(stack_fields(vertex, scale_names))
    theta_raw = stack_fields(vertex, theta_names)
    theta_steps = sigmoid(theta_raw) + theta_residual
    thetas = np.cumsum(theta_steps, axis=-1)
    thetas = thetas / np.maximum(thetas[:, -1:], 1e-8)

    l1l2_rates = sigmoid(stack_fields(vertex, l1l2_names)[:, :1])
    opacity = sigmoid(np.asarray(vertex["opacity"], dtype=np.float32)[:, None])
    acutance = sigmoid(np.asarray(vertex["acutance"], dtype=np.float32)[:, None])
    acutance = acutance * (acutance_max - acutance_min) + acutance_min
    if force_acutance is not None:
        acutance = np.full_like(acutance, float(force_acutance), dtype=np.float32)
    if force_l1l2_rate is not None:
        l1l2_rates = np.full_like(l1l2_rates, float(force_l1l2_rate), dtype=np.float32)
    if force_opacity is not None:
        opacity = np.full_like(opacity, float(force_opacity), dtype=np.float32)
    rotations = quaternion_to_matrix(stack_fields(vertex, rot_names))
    if return_metadata:
        metadata = {
            "acutance_min": float(acutance_min),
            "acutance_max": float(acutance_max),
            "acutance_range_source": acutance_range_source,
            "ply_comments": parse_drk_ply_metadata(ply_data),
        }
        return xyz, colors, scales, thetas, rotations, opacity, acutance, l1l2_rates, metadata
    return xyz, colors, scales, thetas, rotations, opacity, acutance, l1l2_rates


def primitive_scores(opacity, scales=None, acutance=None, score_mode="opacity"):
    if score_mode == "opacity":
        return opacity[:, 0]
    if score_mode == "opacity_scale":
        if scales is None:
            raise ValueError("opacity_scale scoring requires scales")
        return opacity[:, 0] * np.maximum(scales.max(axis=1), 1e-8)
    if score_mode == "opacity_acutance":
        if acutance is None:
            raise ValueError("opacity_acutance scoring requires acutance")
        return opacity[:, 0] * np.maximum(acutance[:, 0], 1e-4)
    raise ValueError(f"Unsupported primitive score mode: {score_mode}")


def select_primitives(opacity, min_opacity, topk, scores=None):
    keep = opacity[:, 0] >= min_opacity
    if topk > 0 and keep.sum() > topk:
        keep_indices = np.nonzero(keep)[0]
        score_values = opacity[:, 0] if scores is None else np.asarray(scores, dtype=np.float32)
        if score_values.shape[0] != opacity.shape[0]:
            raise ValueError("scores must have one value per primitive")
        order = np.argsort(score_values[keep_indices])[::-1][:topk]
        next_keep = np.zeros_like(keep)
        next_keep[keep_indices[order]] = True
        keep = next_keep
    return keep


def apply_scale_filters(
    keep,
    scales,
    max_scale=-1.0,
    max_scale_quantile=-1.0,
    max_scale_area=-1.0,
    max_scale_area_quantile=-1.0,
):
    keep = np.asarray(keep, dtype=bool).copy()
    scales = np.asarray(scales, dtype=np.float32)
    if scales.ndim != 2:
        raise ValueError("scales must have shape [N, K]")
    primitive_max_scale = scales.max(axis=1)
    sorted_scales = np.sort(scales, axis=1)
    primitive_scale_area = sorted_scales[:, -1] * sorted_scales[:, -2]

    def resolve_threshold(values, explicit, quantile, label):
        threshold = None
        explicit = float(explicit)
        quantile = float(quantile)
        if explicit > 0.0:
            threshold = explicit
        if quantile > 0.0:
            if not (0.0 < quantile <= 1.0):
                raise ValueError(f"{label} quantile must be in (0, 1]")
            if not keep.any():
                raise ValueError(f"Cannot apply {label} quantile after all primitives were filtered")
            q_threshold = float(np.quantile(values[keep], quantile))
            threshold = q_threshold if threshold is None else min(threshold, q_threshold)
        return threshold

    scale_threshold = resolve_threshold(
        primitive_max_scale,
        max_scale,
        max_scale_quantile,
        "max scale",
    )
    if scale_threshold is not None:
        keep &= primitive_max_scale <= scale_threshold

    area_threshold = resolve_threshold(
        primitive_scale_area,
        max_scale_area,
        max_scale_area_quantile,
        "max scale area",
    )
    if area_threshold is not None:
        keep &= primitive_scale_area <= area_threshold

    stats = {
        "max_scale": None if float(max_scale) <= 0.0 else float(max_scale),
        "max_scale_quantile": None if float(max_scale_quantile) <= 0.0 else float(max_scale_quantile),
        "max_scale_threshold": scale_threshold,
        "max_scale_area": None if float(max_scale_area) <= 0.0 else float(max_scale_area),
        "max_scale_area_quantile": None if float(max_scale_area_quantile) <= 0.0 else float(max_scale_area_quantile),
        "max_scale_area_threshold": area_threshold,
    }
    return keep, stats


def opacity_boundary_multipliers(opacity, power=0.0, reference=None, min_multiplier=0.25, max_multiplier=4.0):
    opacity_values = np.asarray(opacity, dtype=np.float32).reshape((-1,))
    if opacity_values.size == 0:
        raise ValueError("opacity must contain at least one primitive")
    power = float(power)
    if min_multiplier <= 0.0 or max_multiplier <= 0.0:
        raise ValueError("boundary opacity multiplier limits must be positive")
    if min_multiplier > max_multiplier:
        raise ValueError("boundary opacity multiplier min must be <= max")
    if reference is None or float(reference) <= 0.0:
        reference_value = float(np.clip(opacity_values.mean(), 1e-6, 1.0))
    else:
        reference_value = float(reference)
    if abs(power) < 1e-12:
        return np.ones_like(opacity_values, dtype=np.float32), reference_value
    base = np.clip(opacity_values, 1e-6, 1.0) / max(reference_value, 1e-6)
    multipliers = np.power(base, power).astype(np.float32)
    multipliers = np.clip(multipliers, float(min_multiplier), float(max_multiplier))
    return multipliers, reference_value


def boundary_scale_stats(boundary_alpha, scale_modifier, multipliers):
    base_boundary_scale = math.sqrt(max(-2.0 * math.log(float(boundary_alpha)), 1e-8)) * float(scale_modifier)
    values = base_boundary_scale * np.asarray(multipliers, dtype=np.float32).reshape((-1,))
    return float(values.mean()), float(values.min()), float(values.max())


def primitive_triangle_count(kernel_k, rings, angular_subdivisions):
    rings_arr = np.asarray(rings, dtype=np.int32)
    angular_arr = np.asarray(angular_subdivisions, dtype=np.int32)
    return int(kernel_k) * angular_arr * (2 * rings_arr - 1)


def adaptive_lod_from_scores(
    scores,
    kernel_k,
    rings_min,
    rings_max,
    angular_min,
    angular_max,
    gamma=1.0,
    quantile_low=0.05,
    quantile_high=0.95,
    triangle_budget=-1,
):
    scores = np.asarray(scores, dtype=np.float32).reshape((-1,))
    if scores.size == 0:
        raise ValueError("adaptive LOD requires at least one score")
    if rings_min < 1 or rings_max < 1 or angular_min < 1 or angular_max < 1:
        raise ValueError("adaptive LOD min/max values must be positive")
    if rings_min > rings_max:
        raise ValueError("adaptive rings min must be <= max")
    if angular_min > angular_max:
        raise ValueError("adaptive angular min must be <= max")
    if gamma <= 0.0:
        raise ValueError("adaptive LOD gamma must be positive")
    if not (0.0 <= quantile_low < quantile_high <= 1.0):
        raise ValueError("adaptive LOD quantiles must satisfy 0 <= low < high <= 1")

    finite = np.isfinite(scores)
    if not np.any(finite):
        scores = np.ones_like(scores, dtype=np.float32)
    else:
        min_finite = float(scores[finite].min())
        scores = np.where(finite, scores, min_finite).astype(np.float32)

    lo = float(np.quantile(scores, quantile_low))
    hi = float(np.quantile(scores, quantile_high))
    if hi <= lo + 1e-12:
        normalized = np.ones_like(scores, dtype=np.float32)
    else:
        normalized = np.clip((scores - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    importance = np.power(normalized, float(gamma)).astype(np.float32)

    def lod_for_strength(strength):
        q = np.clip(importance * float(strength), 0.0, 1.0)
        rings = np.rint(float(rings_min) + q * float(rings_max - rings_min)).astype(np.int32)
        angular = np.rint(float(angular_min) + q * float(angular_max - angular_min)).astype(np.int32)
        rings = np.clip(rings, int(rings_min), int(rings_max))
        angular = np.clip(angular, int(angular_min), int(angular_max))
        return rings, angular

    rings, angular = lod_for_strength(1.0)
    target_budget = int(triangle_budget)
    min_triangles = int(np.sum(primitive_triangle_count(kernel_k, rings_min, angular_min)) * scores.size)
    max_triangles = int(np.sum(primitive_triangle_count(kernel_k, rings_max, angular_max)) * scores.size)
    if target_budget > 0:
        if target_budget <= min_triangles:
            rings = np.full(scores.size, int(rings_min), dtype=np.int32)
            angular = np.full(scores.size, int(angular_min), dtype=np.int32)
        elif target_budget >= max_triangles:
            rings = np.full(scores.size, int(rings_max), dtype=np.int32)
            angular = np.full(scores.size, int(angular_max), dtype=np.int32)
        else:
            low_strength = 0.0
            high_strength = 1.0
            for _ in range(24):
                trial_rings, trial_angular = lod_for_strength(high_strength)
                trial_triangles = int(primitive_triangle_count(kernel_k, trial_rings, trial_angular).sum())
                if trial_triangles >= target_budget or high_strength >= 1024.0:
                    break
                high_strength *= 2.0
            best_rings, best_angular = lod_for_strength(low_strength)
            best_triangles = int(primitive_triangle_count(kernel_k, best_rings, best_angular).sum())
            for _ in range(32):
                mid = 0.5 * (low_strength + high_strength)
                trial_rings, trial_angular = lod_for_strength(mid)
                trial_triangles = int(primitive_triangle_count(kernel_k, trial_rings, trial_angular).sum())
                if trial_triangles <= target_budget:
                    best_rings, best_angular = trial_rings, trial_angular
                    best_triangles = trial_triangles
                    low_strength = mid
                else:
                    high_strength = mid
            rings, angular = best_rings, best_angular

    triangles = primitive_triangle_count(kernel_k, rings, angular).astype(np.int64)
    stats = {
        "adaptive_lod": True,
        "lod_score_min": float(scores.min()),
        "lod_score_max": float(scores.max()),
        "lod_score_mean": float(scores.mean()),
        "lod_score_quantile_low": float(lo),
        "lod_score_quantile_high": float(hi),
        "lod_gamma": float(gamma),
        "lod_triangle_budget": target_budget,
        "lod_triangle_budget_min": int(min_triangles),
        "lod_triangle_budget_max": int(max_triangles),
        "rings_min": int(rings.min()),
        "rings_max": int(rings.max()),
        "rings_mean": float(rings.mean()),
        "angular_subdivisions_min": int(angular.min()),
        "angular_subdivisions_max": int(angular.max()),
        "angular_subdivisions_mean": float(angular.mean()),
        "triangles_min": int(triangles.min()),
        "triangles_max": int(triangles.max()),
        "triangles_mean": float(triangles.mean()),
        "triangles_total": int(triangles.sum()),
    }
    return rings, angular, stats


def sharpen_kernel_opacity(kernel_opacity, acutance):
    hard = acutance >= 1.0 - 1e-6
    k_acu = np.minimum(acutance, 0.999999)
    cond1 = kernel_opacity < ((1.0 + k_acu) / 4.0)
    cond2 = kernel_opacity < ((3.0 - k_acu) / 4.0)
    middle = (1.0 + k_acu) / np.maximum(1.0 - k_acu, 1e-8) * kernel_opacity - k_acu / np.maximum(1.0 - k_acu, 1e-8)
    low = (1.0 - k_acu) / (1.0 + k_acu) * kernel_opacity
    high = (1.0 - k_acu) / (1.0 + k_acu) * kernel_opacity + 2.0 * k_acu / (1.0 + k_acu)
    soft = np.where(cond2, np.where(cond1, low, middle), high)
    return np.where(hard, kernel_opacity >= 0.5, soft)


def build_triangle_mesh(
    xyz,
    colors,
    scales,
    thetas,
    rotations,
    boundary_alpha,
    scale_modifier,
    opacity=None,
    acutance=None,
    rings=1,
    angular_subdivisions=1,
    boundary_scale_multiplier=None,
    return_float_colors=False,
):
    if not (0.0 < boundary_alpha < 1.0):
        raise ValueError("boundary_alpha must be in (0, 1)")
    kernel_k = scales.shape[1]
    primitive_count = xyz.shape[0]
    if boundary_scale_multiplier is None:
        boundary_scale_multiplier = np.ones((primitive_count,), dtype=np.float32)
    else:
        boundary_scale_multiplier = np.asarray(boundary_scale_multiplier, dtype=np.float32).reshape((-1,))
        if boundary_scale_multiplier.shape[0] != primitive_count:
            raise ValueError("boundary_scale_multiplier must have one value per primitive")
        if not np.all(np.isfinite(boundary_scale_multiplier)) or np.any(boundary_scale_multiplier <= 0.0):
            raise ValueError("boundary_scale_multiplier values must be finite and positive")
    rings_values = np.asarray(rings)
    angular_values = np.asarray(angular_subdivisions)
    if rings_values.ndim > 0 or angular_values.ndim > 0:
        if rings_values.ndim == 0:
            rings_array = np.full(primitive_count, int(rings), dtype=np.int32)
        else:
            rings_array = rings_values.astype(np.int32).reshape((-1,))
        if angular_values.ndim == 0:
            angular_array = np.full(primitive_count, int(angular_subdivisions), dtype=np.int32)
        else:
            angular_array = angular_values.astype(np.int32).reshape((-1,))
        if rings_array.shape[0] != primitive_count:
            raise ValueError("rings array must have one value per primitive")
        if angular_array.shape[0] != primitive_count:
            raise ValueError("angular_subdivisions array must have one value per primitive")
        if np.any(rings_array < 1):
            raise ValueError("rings values must be positive")
        if np.any(angular_array < 1):
            raise ValueError("angular_subdivisions values must be positive")

        vertices_parts = []
        color_parts = []
        float_color_parts = []
        face_parts = []
        vertex_offset = 0
        unique_lods = sorted({(int(r), int(a)) for r, a in zip(rings_array.tolist(), angular_array.tolist())})
        for lod_rings, lod_angular in unique_lods:
            lod_mask = (rings_array == lod_rings) & (angular_array == lod_angular)
            if not np.any(lod_mask):
                continue
            if return_float_colors:
                vertices_lod, colors_lod, faces_lod, _, float_colors_lod = build_triangle_mesh(
                    xyz[lod_mask],
                    colors[lod_mask],
                    scales[lod_mask],
                    thetas[lod_mask],
                    rotations[lod_mask],
                    boundary_alpha,
                    scale_modifier,
                    opacity=opacity[lod_mask] if opacity is not None else None,
                    acutance=acutance[lod_mask] if acutance is not None else None,
                    rings=lod_rings,
                    angular_subdivisions=lod_angular,
                    boundary_scale_multiplier=boundary_scale_multiplier[lod_mask],
                    return_float_colors=True,
                )
                float_color_parts.append(float_colors_lod)
            else:
                vertices_lod, colors_lod, faces_lod, _ = build_triangle_mesh(
                    xyz[lod_mask],
                    colors[lod_mask],
                    scales[lod_mask],
                    thetas[lod_mask],
                    rotations[lod_mask],
                    boundary_alpha,
                    scale_modifier,
                    opacity=opacity[lod_mask] if opacity is not None else None,
                    acutance=acutance[lod_mask] if acutance is not None else None,
                    rings=lod_rings,
                    angular_subdivisions=lod_angular,
                    boundary_scale_multiplier=boundary_scale_multiplier[lod_mask],
                )
            vertices_parts.append(vertices_lod)
            color_parts.append(colors_lod)
            face_parts.append(faces_lod + vertex_offset)
            vertex_offset += vertices_lod.shape[0]
        base_boundary_scale = math.sqrt(max(-2.0 * math.log(boundary_alpha), 1e-8))
        boundary_scale = float((base_boundary_scale * scale_modifier * boundary_scale_multiplier).mean())
        result = (
            np.concatenate(vertices_parts, axis=0),
            np.concatenate(color_parts, axis=0),
            np.concatenate(face_parts, axis=0),
            boundary_scale,
        )
        if return_float_colors:
            return result + (np.concatenate(float_color_parts, axis=0),)
        return result

    rings = int(rings)
    angular_subdivisions = int(angular_subdivisions)
    if rings < 1:
        raise ValueError("rings must be positive")
    if angular_subdivisions < 1:
        raise ValueError("angular_subdivisions must be positive")
    angular_count = kernel_k * angular_subdivisions
    base_boundary_scale = math.sqrt(max(-2.0 * math.log(boundary_alpha), 1e-8))
    boundary_scale = float((base_boundary_scale * scale_modifier * boundary_scale_multiplier).mean())
    if thetas.shape != scales.shape:
        raise ValueError("scales and thetas must have matching shapes")

    # CUDA builds kernel vertex i at the left boundary of segment i:
    # 0, theta_0, ..., theta_{K-2}. theta_{K-1} is the closing 2pi endpoint.
    vertex_thetas = np.concatenate(
        [np.zeros_like(thetas[:, :1]), thetas[:, :-1]],
        axis=1,
    )
    angles = vertex_thetas * (2.0 * math.pi)
    base_kernel_vertices = np.stack(
        [
            np.cos(angles) * scales,
            np.sin(angles) * scales,
        ],
        axis=-1,
    )
    subdivided_kernel_vertices = np.empty((scales.shape[0], angular_count, 2), dtype=np.float32)
    for corner in range(kernel_k):
        left = base_kernel_vertices[:, corner, :]
        right = base_kernel_vertices[:, (corner + 1) % kernel_k, :]
        for subdivision in range(angular_subdivisions):
            rate = subdivision / float(angular_subdivisions)
            subdivided_kernel_vertices[:, corner * angular_subdivisions + subdivision, :] = (
                (1.0 - rate) * left + rate * right
            )
    base_ring_scale_factors = np.linspace(
        base_boundary_scale / rings,
        base_boundary_scale,
        rings,
        dtype=np.float32,
    )
    ring_scale_factors = base_ring_scale_factors[None, :] * boundary_scale_multiplier[:, None]
    ring_kernel_alpha = np.exp(-0.5 * np.square(ring_scale_factors)).astype(np.float32)
    local_xy = (
        subdivided_kernel_vertices[:, None, :, :]
        * ring_scale_factors[:, :, None, None]
        * scale_modifier
    )
    local = np.concatenate(
        [
            local_xy,
            np.zeros((scales.shape[0], rings, angular_count, 1), dtype=np.float32),
        ],
        axis=-1,
    )
    ring_vertices = xyz[:, None, None, :] + np.einsum("nij,nrkj->nrki", rotations, local)

    vertices_per_primitive = 1 + rings * angular_count
    triangles_per_primitive = angular_count + max(0, rings - 1) * angular_count * 2
    vertices = np.empty((primitive_count * vertices_per_primitive, 3), dtype=np.float32)
    color_channels = 4 if opacity is not None else 3
    vertex_colors = np.empty((primitive_count * vertices_per_primitive, color_channels), dtype=np.uint8)
    vertex_colors_float = (
        np.empty((primitive_count * vertices_per_primitive, color_channels), dtype=np.float32)
        if return_float_colors
        else None
    )
    faces = np.empty((primitive_count * triangles_per_primitive, 3), dtype=np.int32)

    color_float = np.clip(colors, 0.0, 1.0).astype(np.float32, copy=False)
    color_u8 = np.round(color_float * 255.0).astype(np.uint8)
    alpha = None
    alpha_u8 = None
    if opacity is not None:
        if acutance is None or rings == 1:
            alpha = np.repeat(opacity[:, :1], vertices_per_primitive, axis=1)
        else:
            center_alpha = opacity[:, :1]
            ring_sharpen = sharpen_kernel_opacity(ring_kernel_alpha, acutance[:, :1])
            ring_alpha = opacity[:, :1, None] * ring_sharpen[:, :, None]
            ring_alpha = np.repeat(ring_alpha, angular_count, axis=2)
            alpha = np.concatenate(
                [center_alpha, ring_alpha.reshape((primitive_count, rings * angular_count))],
                axis=1,
            )
        alpha_u8 = np.round(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    for idx in range(primitive_count):
        vertex_base = idx * vertices_per_primitive
        face_base = idx * triangles_per_primitive
        vertices[vertex_base] = xyz[idx]
        vertices[vertex_base + 1 : vertex_base + vertices_per_primitive] = ring_vertices[idx].reshape((rings * angular_count, 3))
        vertex_colors[vertex_base : vertex_base + vertices_per_primitive, :3] = color_u8[idx]
        if vertex_colors_float is not None:
            vertex_colors_float[vertex_base : vertex_base + vertices_per_primitive, :3] = color_float[idx]
        if alpha_u8 is not None:
            vertex_colors[vertex_base : vertex_base + vertices_per_primitive, 3] = alpha_u8[idx]
            if vertex_colors_float is not None:
                vertex_colors_float[vertex_base : vertex_base + vertices_per_primitive, 3] = alpha[idx]
        for corner in range(angular_count):
            faces[face_base + corner] = [
                vertex_base,
                vertex_base + 1 + corner,
                vertex_base + 1 + ((corner + 1) % angular_count),
            ]
        write_face = face_base + angular_count
        for ring in range(1, rings):
            prev_base = vertex_base + 1 + (ring - 1) * angular_count
            curr_base = vertex_base + 1 + ring * angular_count
            for corner in range(angular_count):
                prev0 = prev_base + corner
                prev1 = prev_base + ((corner + 1) % angular_count)
                curr0 = curr_base + corner
                curr1 = curr_base + ((corner + 1) % angular_count)
                faces[write_face] = [prev0, curr0, curr1]
                faces[write_face + 1] = [prev0, curr1, prev1]
                write_face += 2
    if return_float_colors:
        return vertices, vertex_colors, faces, boundary_scale, vertex_colors_float
    return vertices, vertex_colors, faces, boundary_scale


def write_mesh_ply(
    path,
    vertices,
    colors,
    faces,
    comments=None,
    face_primitive_indices=None,
    alpha_float=None,
    rgb_float=None,
):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if colors.shape[1] not in {3, 4}:
        raise ValueError("colors must have shape [N, 3] or [N, 4]")
    if rgb_float is not None:
        rgb_float = np.asarray(rgb_float, dtype=np.float32)
        if rgb_float.shape != (colors.shape[0], 3):
            raise ValueError("rgb_float must have shape [N, 3]")
    if alpha_float is not None:
        alpha_float = np.asarray(alpha_float, dtype=np.float32).reshape((-1,))
        if alpha_float.shape[0] != colors.shape[0]:
            raise ValueError("alpha_float must have one value per vertex")
    vertex_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    if rgb_float is not None:
        vertex_dtype.extend(
            [
                ("red_float", "f4"),
                ("green_float", "f4"),
                ("blue_float", "f4"),
            ]
        )
    if colors.shape[1] == 4:
        vertex_dtype.append(("alpha", "u1"))
    if alpha_float is not None:
        vertex_dtype.append(("alpha_float", "f4"))
    vertex_data = np.empty(vertices.shape[0], dtype=vertex_dtype)
    vertex_data["x"] = vertices[:, 0]
    vertex_data["y"] = vertices[:, 1]
    vertex_data["z"] = vertices[:, 2]
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]
    if rgb_float is not None:
        vertex_data["red_float"] = rgb_float[:, 0]
        vertex_data["green_float"] = rgb_float[:, 1]
        vertex_data["blue_float"] = rgb_float[:, 2]
    if colors.shape[1] == 4:
        vertex_data["alpha"] = colors[:, 3]
    if alpha_float is not None:
        vertex_data["alpha_float"] = np.clip(alpha_float, 0.0, 1.0)

    face_dtype = [("vertex_indices", "i4", (3,))]
    if face_primitive_indices is not None:
        face_primitive_indices = np.asarray(face_primitive_indices, dtype=np.int32).reshape((-1,))
        if face_primitive_indices.shape[0] != faces.shape[0]:
            raise ValueError("face_primitive_indices must have one value per face")
        face_dtype.append(("primitive_index", "i4"))
    face_data = np.empty(faces.shape[0], dtype=face_dtype)
    face_data["vertex_indices"] = faces
    if face_primitive_indices is not None:
        face_data["primitive_index"] = face_primitive_indices
    PlyData(
        [
            PlyElement.describe(vertex_data, "vertex"),
            PlyElement.describe(face_data, "face"),
        ],
        comments=comments,
    ).write(path)


def read_mesh_ply(path, return_face_properties=False):
    ply_data = PlyData.read(path)
    vertex = ply_data["vertex"].data
    names = vertex.dtype.names
    required = ["x", "y", "z", "red", "green", "blue"]
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"{path} is missing mesh vertex fields: {', '.join(missing)}")
    vertices = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )
    color_parts = [
        np.asarray(vertex["red"], dtype=np.uint8),
        np.asarray(vertex["green"], dtype=np.uint8),
        np.asarray(vertex["blue"], dtype=np.uint8),
    ]
    if "alpha" in names:
        color_parts.append(np.asarray(vertex["alpha"], dtype=np.uint8))
    colors = np.stack(color_parts, axis=1)

    raw_face = ply_data["face"].data
    face_data = raw_face["vertex_indices"]
    faces = np.stack([np.asarray(face, dtype=np.int32) for face in face_data], axis=0)
    if faces.shape[1] != 3:
        raise ValueError(f"{path} contains non-triangle faces; got face shape {faces.shape}")
    face_properties = {}
    if "primitive_index" in raw_face.dtype.names:
        face_properties["primitive_index"] = np.asarray(raw_face["primitive_index"], dtype=np.int32)
    if all(name in names for name in ("red_float", "green_float", "blue_float")):
        face_properties["vertex_rgb_float"] = np.stack(
            [
                np.asarray(vertex["red_float"], dtype=np.float32),
                np.asarray(vertex["green_float"], dtype=np.float32),
                np.asarray(vertex["blue_float"], dtype=np.float32),
            ],
            axis=1,
        )
    if "alpha_float" in names:
        face_properties["vertex_alpha_float"] = np.asarray(vertex["alpha_float"], dtype=np.float32)
    embedded_center_names = ("drk_sh_center_x", "drk_sh_center_y", "drk_sh_center_z")
    embedded_coeff_count = embedded_sh_coeff_count(names)
    if all(name in names for name in embedded_center_names) and embedded_coeff_count > 0:
        face_properties["vertex_sh_primitive_xyz"] = np.stack(
            [np.asarray(vertex[name], dtype=np.float32) for name in embedded_center_names],
            axis=1,
        )
        face_properties["vertex_sh_features"] = np.stack(
            [
                np.stack(
                    [
                        np.asarray(vertex[f"drk_sh_{coeff_idx}_{channel}"], dtype=np.float32)
                        for channel in ("r", "g", "b")
                    ],
                    axis=1,
                )
                for coeff_idx in range(embedded_coeff_count)
            ],
            axis=2,
        ).astype(np.float32, copy=False)

    primitive_element = optional_ply_element(ply_data, "drk_sh_primitive")
    if primitive_element is not None:
        primitive_data = primitive_element.data
        primitive_names = primitive_data.dtype.names
        primitive_coeff_count = embedded_sh_coeff_count(primitive_names)
        if all(name in primitive_names for name in embedded_center_names) and primitive_coeff_count > 0:
            if "primitive_index" in primitive_names:
                face_properties["primitive_sh_primitive_index"] = np.asarray(
                    primitive_data["primitive_index"],
                    dtype=np.int32,
                )
            face_properties["primitive_sh_primitive_xyz"] = np.stack(
                [np.asarray(primitive_data[name], dtype=np.float32) for name in embedded_center_names],
                axis=1,
            )
            face_properties["primitive_sh_features"] = np.stack(
                [
                    np.stack(
                        [
                            np.asarray(primitive_data[f"drk_sh_{coeff_idx}_{channel}"], dtype=np.float32)
                            for channel in ("r", "g", "b")
                        ],
                        axis=1,
                    )
                    for coeff_idx in range(primitive_coeff_count)
                ],
                axis=2,
            ).astype(np.float32, copy=False)

    metadata = parse_drk_ply_metadata(ply_data)
    stats = {
        "mesh_input": os.path.abspath(path),
        "mesh_output": os.path.abspath(path),
        "checkpoint_ply": metadata.get("drk_source"),
        "input_primitives": metadata_int(metadata, "drk_input_primitives"),
        "exported_primitives": metadata_int(metadata, "drk_exported_primitives"),
        "kernel_K": metadata_int(metadata, "drk_kernel_K"),
        "max_scale": metadata_float(metadata, "drk_max_scale"),
        "max_scale_quantile": metadata_float(metadata, "drk_max_scale_quantile"),
        "max_scale_threshold": metadata_float(metadata, "drk_max_scale_threshold"),
        "max_scale_area": metadata_float(metadata, "drk_max_scale_area"),
        "max_scale_area_quantile": metadata_float(metadata, "drk_max_scale_area_quantile"),
        "max_scale_area_threshold": metadata_float(metadata, "drk_max_scale_area_threshold"),
        "vertices": int(vertices.shape[0]),
        "triangles": int(faces.shape[0]),
        "acutance_min": metadata_float(metadata, "drk_acutance_min"),
        "acutance_max": metadata_float(metadata, "drk_acutance_max"),
        "acutance_range_source": metadata.get("drk_acutance_range_source"),
        "topk_score": metadata.get("drk_topk_score"),
        "color_mode": metadata.get("drk_mesh_color_mode"),
        "view_dependent_colors": metadata_bool(metadata, "drk_mesh_view_dependent_colors"),
        "color_bake_split": metadata.get("drk_mesh_color_bake_split"),
        "color_bake_views": metadata_int(metadata, "drk_mesh_color_bake_views"),
        "color_sample_align_corners": metadata_bool(metadata, "drk_mesh_color_sample_align_corners"),
        "color_sample_weight": metadata.get("drk_mesh_color_sample_weight"),
        "color_sample_space": metadata.get("drk_mesh_color_sample_space"),
        "color_sample_reducer": metadata.get("drk_mesh_color_sample_reducer"),
        "color_sample_trim_fraction": metadata_float(metadata, "drk_mesh_color_sample_trim_fraction"),
        "color_depth_filter": metadata_bool(metadata, "drk_mesh_color_depth_filter"),
        "color_depth_abs_tol": metadata_float(metadata, "drk_mesh_color_depth_abs_tol"),
        "color_depth_rel_tol": metadata_float(metadata, "drk_mesh_color_depth_rel_tol"),
        "mesh_sh_sidecar": metadata.get("drk_mesh_sh_sidecar"),
        "mesh_sh_sidecar_sha256": metadata.get("drk_mesh_sh_sidecar_sha256"),
        "mesh_sh_embedded": metadata_bool(metadata, "drk_mesh_sh_embedded"),
        "mesh_sh_embedded_format": metadata.get("drk_mesh_sh_embedded_format"),
        "mesh_sh_embedded_sh_degree": metadata_int(metadata, "drk_mesh_sh_embedded_sh_degree"),
        "mesh_sh_embedded_render_sh_degree": metadata_int(metadata, "drk_mesh_sh_embedded_render_sh_degree"),
        "mesh_sh_embedded_source_sh_degree": metadata_int(metadata, "drk_mesh_sh_embedded_source_sh_degree"),
        "mesh_sh_embedded_coeff_count": metadata_int(metadata, "drk_mesh_sh_embedded_coeff_count"),
        "mesh_asset_ply_format": metadata.get("drk_mesh_asset_ply_format"),
        "mesh_asset_self_contained": metadata_bool(metadata, "drk_mesh_asset_self_contained"),
        "mesh_asset_source_paths_omitted": metadata_bool(metadata, "drk_mesh_asset_source_paths_omitted"),
        "boundary_alpha": metadata_float(metadata, "drk_boundary_alpha"),
        "scale_modifier": metadata_float(metadata, "drk_scale_modifier"),
        "support_alpha": metadata_float(metadata, "drk_support_alpha"),
        "boundary_scale": metadata_float(metadata, "drk_boundary_scale"),
        "boundary_scale_min": metadata_float(metadata, "drk_boundary_scale_min"),
        "boundary_scale_max": metadata_float(metadata, "drk_boundary_scale_max"),
        "boundary_opacity_power": metadata_float(metadata, "drk_boundary_opacity_power"),
        "boundary_opacity_reference": metadata_float(metadata, "drk_boundary_opacity_reference"),
        "mesh_alpha_mode": metadata.get("drk_mesh_alpha_mode", "opacity" if colors.shape[1] == 4 else "opaque"),
        "adaptive_lod": metadata_bool(metadata, "drk_mesh_adaptive_lod"),
        "lod_score": metadata.get("drk_mesh_lod_score"),
        "lod_gamma": metadata_float(metadata, "drk_mesh_lod_gamma"),
        "lod_triangle_budget": metadata_int(metadata, "drk_mesh_lod_triangle_budget"),
        "rings": metadata_int(metadata, "drk_mesh_rings"),
        "rings_min": metadata_int(metadata, "drk_mesh_rings_min"),
        "rings_max": metadata_int(metadata, "drk_mesh_rings_max"),
        "rings_mean": metadata_float(metadata, "drk_mesh_rings_mean"),
        "angular_subdivisions": metadata_int(metadata, "drk_mesh_angular_subdivisions"),
        "angular_subdivisions_min": metadata_int(metadata, "drk_mesh_angular_subdivisions_min"),
        "angular_subdivisions_max": metadata_int(metadata, "drk_mesh_angular_subdivisions_max"),
        "angular_subdivisions_mean": metadata_float(metadata, "drk_mesh_angular_subdivisions_mean"),
        "flat_face_colors": metadata_bool(metadata, "drk_mesh_flat_face_colors"),
        "has_vertex_alpha": bool(colors.shape[1] == 4),
        "has_vertex_rgb_float": bool("vertex_rgb_float" in face_properties),
        "has_vertex_alpha_float": bool("alpha_float" in names),
        "has_face_primitive_indices": bool("primitive_index" in face_properties),
    }
    if "vertex_sh_features" in face_properties:
        stats["mesh_sh_embedded"] = True
        stats["mesh_sh_embedded_coeff_count"] = int(face_properties["vertex_sh_features"].shape[2])
    if "primitive_sh_features" in face_properties:
        stats["mesh_sh_embedded"] = True
        stats["mesh_sh_embedded_coeff_count"] = int(face_properties["primitive_sh_features"].shape[2])
    if stats.get("adaptive_lod"):
        if stats.get("rings") == 0:
            stats["rings"] = None
        if stats.get("angular_subdivisions") == 0:
            stats["angular_subdivisions"] = None
    if return_face_properties:
        return vertices, colors, faces, stats, face_properties
    return vertices, colors, faces, stats


def mesh_colors_to_float(colors, face_properties=None):
    colors = np.asarray(colors)
    out = colors.astype(np.float32) / 255.0
    if face_properties is not None and "vertex_rgb_float" in face_properties:
        rgb = np.asarray(face_properties["vertex_rgb_float"], dtype=np.float32)
        if rgb.shape != (out.shape[0], 3):
            raise ValueError("vertex_rgb_float must have shape [N, 3]")
        out[:, :3] = rgb
    if face_properties is None or "vertex_alpha_float" not in face_properties:
        return out
    alpha = np.asarray(face_properties["vertex_alpha_float"], dtype=np.float32).reshape((-1,))
    if alpha.shape[0] != out.shape[0]:
        raise ValueError("vertex_alpha_float must have one value per mesh vertex")
    alpha = np.clip(alpha, 0.0, 1.0)[:, None]
    if out.shape[1] >= 4:
        out[:, 3:4] = alpha
        return out
    return np.concatenate([out[:, :3], alpha], axis=1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a trained DRK point_cloud.ply into a colored triangle mesh."
    )
    parser.add_argument("--input", "-i", required=True, help="Input DRK point_cloud.ply")
    parser.add_argument("--output", "-o", required=True, help="Output colored triangle PLY")
    parser.add_argument("--summary", default=None, help="Optional JSON export summary path")
    parser.add_argument("--min-opacity", type=float, default=0.01, help="Opacity cutoff before export")
    parser.add_argument("--topk", type=int, default=-1, help="Keep only the top-K opacities after cutoff")
    parser.add_argument(
        "--topk-score",
        choices=["opacity", "opacity_scale", "opacity_acutance"],
        default="opacity",
        help="Primitive score used when --topk is positive",
    )
    parser.add_argument(
        "--max-scale",
        type=float,
        default=-1.0,
        help="Optional activated max-scale cutoff for rejecting mesh-only fan outliers; <=0 disables it.",
    )
    parser.add_argument(
        "--max-scale-quantile",
        type=float,
        default=-1.0,
        help="Optional quantile cutoff on activated max scale after opacity/top-k filtering; <=0 disables it.",
    )
    parser.add_argument(
        "--max-scale-area",
        type=float,
        default=-1.0,
        help="Optional cutoff on product of the two largest activated scales; <=0 disables it.",
    )
    parser.add_argument(
        "--max-scale-area-quantile",
        type=float,
        default=-1.0,
        help="Optional quantile cutoff on product of the two largest activated scales; <=0 disables it.",
    )
    parser.add_argument(
        "--boundary-alpha",
        type=float,
        default=0.5,
        help="DRK kernel opacity level to treat as the exported L1 boundary",
    )
    parser.add_argument(
        "--boundary-opacity-power",
        type=float,
        default=0.0,
        help="Opt-in per-primitive boundary multiplier: (opacity / reference) ** power; 0 preserves the global boundary.",
    )
    parser.add_argument(
        "--boundary-opacity-reference",
        type=float,
        default=-1.0,
        help="Reference opacity for --boundary-opacity-power; <=0 uses the exported primitive mean opacity.",
    )
    parser.add_argument("--boundary-scale-min", type=float, default=0.25, help="Minimum per-primitive boundary multiplier")
    parser.add_argument("--boundary-scale-max", type=float, default=4.0, help="Maximum per-primitive boundary multiplier")
    parser.add_argument("--scale-modifier", type=float, default=1.0, help="Uniform scale multiplier")
    parser.add_argument(
        "--support-alpha",
        type=float,
        default=None,
        help="Crop support to the equivalent unsharpened 3-sigma tail alpha; mutually exclusive with non-default --scale-modifier.",
    )
    parser.add_argument("--include-alpha", action="store_true", help="Write DRK opacity as vertex alpha")
    parser.add_argument(
        "--rings",
        type=int,
        default=1,
        help="Number of radial triangle rings per primitive; values >1 approximate the DRK opacity profile",
    )
    parser.add_argument(
        "--angular-subdivisions",
        type=int,
        default=1,
        help="Number of linear subdivisions per DRK angular wedge for denser vertex-color baking",
    )
    parser.add_argument(
        "--acutance-min",
        type=float,
        default=None,
        help="Acutance activation minimum used by the saved checkpoint; defaults to PLY metadata, then -0.1",
    )
    parser.add_argument(
        "--acutance-max",
        type=float,
        default=None,
        help="Acutance activation maximum used by the saved checkpoint; defaults to PLY metadata, then 0.99",
    )
    parser.add_argument("--force-acutance", type=float, default=None, help="Export as if every DRK primitive has this acutance value.")
    parser.add_argument("--force-l1l2-rate", type=float, default=None, help="Export as if every DRK primitive has this L1/L2 interpolation rate.")
    parser.add_argument("--force-opacity", type=float, default=None, help="Export as if every DRK primitive has this opacity value.")
    parser.add_argument("--theta-residual", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not (0.0 < args.boundary_alpha < 1.0):
        raise ValueError("--boundary-alpha must be in (0, 1)")
    if args.rings < 1:
        raise ValueError("--rings must be positive")
    if args.angular_subdivisions < 1:
        raise ValueError("--angular-subdivisions must be positive")
    if args.boundary_scale_min <= 0.0 or args.boundary_scale_max <= 0.0:
        raise ValueError("--boundary-scale-min/max must be positive")
    if args.boundary_scale_min > args.boundary_scale_max:
        raise ValueError("--boundary-scale-min must be <= --boundary-scale-max")
    args.scale_modifier = resolve_scale_modifier(args.scale_modifier, args.support_alpha)

    xyz, colors, scales, thetas, rotations, opacity, acutance, l1l2_rates, metadata = load_drk(
        args.input,
        args.acutance_min,
        args.acutance_max,
        args.theta_residual,
        return_metadata=True,
        force_acutance=args.force_acutance,
        force_l1l2_rate=args.force_l1l2_rate,
        force_opacity=args.force_opacity,
    )
    scores = primitive_scores(opacity, scales=scales, acutance=acutance, score_mode=args.topk_score)
    keep = select_primitives(opacity, args.min_opacity, args.topk, scores=scores)
    keep, scale_filter_stats = apply_scale_filters(
        keep,
        scales,
        args.max_scale,
        args.max_scale_quantile,
        args.max_scale_area,
        args.max_scale_area_quantile,
    )
    if keep.sum() == 0:
        raise ValueError("No DRK primitives survived the export filters")
    boundary_multipliers, boundary_opacity_reference = opacity_boundary_multipliers(
        opacity[keep],
        args.boundary_opacity_power,
        args.boundary_opacity_reference,
        args.boundary_scale_min,
        args.boundary_scale_max,
    )
    boundary_scale_mean, boundary_scale_min, boundary_scale_max = boundary_scale_stats(
        args.boundary_alpha,
        args.scale_modifier,
        boundary_multipliers,
    )

    vertices, vertex_colors, faces, boundary_scale, vertex_colors_float = build_triangle_mesh(
        xyz[keep],
        colors[keep],
        scales[keep],
        thetas[keep],
        rotations[keep],
        args.boundary_alpha,
        args.scale_modifier,
        opacity=opacity[keep] if args.include_alpha else None,
        acutance=acutance[keep],
        rings=args.rings,
        angular_subdivisions=args.angular_subdivisions,
        boundary_scale_multiplier=boundary_multipliers,
        return_float_colors=True,
    )
    comments = [
        f"drk_source {os.path.abspath(args.input)}",
        f"drk_kernel_K {int(scales.shape[1])}",
        f"drk_acutance_min {metadata['acutance_min']:.9g}",
        f"drk_acutance_max {metadata['acutance_max']:.9g}",
        f"drk_acutance_range_source {metadata['acutance_range_source']}",
        f"drk_input_primitives {int(xyz.shape[0])}",
        f"drk_exported_primitives {int(keep.sum())}",
        f"drk_max_scale {'' if scale_filter_stats['max_scale'] is None else float(scale_filter_stats['max_scale']):}",
        f"drk_max_scale_quantile {'' if scale_filter_stats['max_scale_quantile'] is None else float(scale_filter_stats['max_scale_quantile']):}",
        f"drk_max_scale_threshold {'' if scale_filter_stats['max_scale_threshold'] is None else float(scale_filter_stats['max_scale_threshold']):}",
        f"drk_max_scale_area {'' if scale_filter_stats['max_scale_area'] is None else float(scale_filter_stats['max_scale_area']):}",
        f"drk_max_scale_area_quantile {'' if scale_filter_stats['max_scale_area_quantile'] is None else float(scale_filter_stats['max_scale_area_quantile']):}",
        f"drk_max_scale_area_threshold {'' if scale_filter_stats['max_scale_area_threshold'] is None else float(scale_filter_stats['max_scale_area_threshold']):}",
        f"drk_boundary_alpha {float(args.boundary_alpha):.9g}",
        f"drk_scale_modifier {float(args.scale_modifier):.9g}",
        f"drk_boundary_scale {float(boundary_scale):.9g}",
        f"drk_boundary_scale_min {float(boundary_scale_min):.9g}",
        f"drk_boundary_scale_max {float(boundary_scale_max):.9g}",
        f"drk_boundary_opacity_power {float(args.boundary_opacity_power):.9g}",
        f"drk_boundary_opacity_reference {float(boundary_opacity_reference):.9g}",
        f"drk_force_acutance {'' if args.force_acutance is None else float(args.force_acutance):}",
        f"drk_force_l1l2_rate {'' if args.force_l1l2_rate is None else float(args.force_l1l2_rate):}",
        f"drk_force_opacity {'' if args.force_opacity is None else float(args.force_opacity):}",
        f"drk_mesh_alpha_mode {'opacity' if args.include_alpha else 'opaque'}",
        f"drk_mesh_rings {int(args.rings)}",
        f"drk_mesh_angular_subdivisions {int(args.angular_subdivisions)}",
        f"drk_topk_score {args.topk_score}",
        "drk_mesh_color_mode dc",
    ]
    if args.support_alpha is not None:
        comments.insert(9, f"drk_support_alpha {float(args.support_alpha):.9g}")
    # Emit per-face source-primitive indices so the mesh is self-describing and
    # directly usable by the static-color fitter / SH-view bakers (they map each
    # face back to its DRK primitive). Faces are built per primitive in order,
    # so each kept primitive owns a fixed contiguous block of faces.
    kept_indices = np.flatnonzero(keep).astype(np.int32)
    n_kept = int(kept_indices.shape[0])
    face_primitive_indices = None
    if n_kept and faces.shape[0] % n_kept == 0:
        faces_per_primitive = faces.shape[0] // n_kept
        face_primitive_indices = np.repeat(kept_indices, faces_per_primitive)
    write_mesh_ply(
        args.output,
        vertices,
        vertex_colors,
        faces,
        comments=comments,
        alpha_float=vertex_colors_float[:, 3] if vertex_colors_float.shape[1] == 4 else None,
        face_primitive_indices=face_primitive_indices,
    )

    stats = ExportStats(
        input_path=os.path.abspath(args.input),
        output_path=os.path.abspath(args.output),
        input_primitives=int(xyz.shape[0]),
        exported_primitives=int(keep.sum()),
        kernel_k=int(scales.shape[1]),
        vertices=int(vertices.shape[0]),
        triangles=int(faces.shape[0]),
        min_opacity=float(args.min_opacity),
        max_scale=scale_filter_stats["max_scale"],
        max_scale_quantile=scale_filter_stats["max_scale_quantile"],
        max_scale_threshold=scale_filter_stats["max_scale_threshold"],
        max_scale_area=scale_filter_stats["max_scale_area"],
        max_scale_area_quantile=scale_filter_stats["max_scale_area_quantile"],
        max_scale_area_threshold=scale_filter_stats["max_scale_area_threshold"],
        mean_opacity=float(opacity[keep].mean()),
        mean_acutance=float(acutance[keep].mean()),
        mean_l1l2_rate=float(l1l2_rates[keep].mean()),
        acutance_min=float(metadata["acutance_min"]),
        acutance_max=float(metadata["acutance_max"]),
        acutance_range_source=metadata["acutance_range_source"],
        topk_score=args.topk_score,
        boundary_alpha=float(args.boundary_alpha),
        scale_modifier=float(args.scale_modifier),
        support_alpha=None if args.support_alpha is None else float(args.support_alpha),
        boundary_scale=float(boundary_scale_mean),
        boundary_scale_min=float(boundary_scale_min),
        boundary_scale_max=float(boundary_scale_max),
        boundary_opacity_power=float(args.boundary_opacity_power),
        boundary_opacity_reference=float(boundary_opacity_reference),
        force_acutance=None if args.force_acutance is None else float(args.force_acutance),
        force_l1l2_rate=None if args.force_l1l2_rate is None else float(args.force_l1l2_rate),
        force_opacity=None if args.force_opacity is None else float(args.force_opacity),
        rings=int(args.rings),
        angular_subdivisions=int(args.angular_subdivisions),
        has_vertex_alpha=bool(vertex_colors.shape[1] == 4),
    )

    summary_path = args.summary
    if summary_path is None:
        summary_path = os.path.splitext(args.output)[0] + "_summary.json"
    os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(asdict(stats), f, indent=2, sort_keys=True)
    print(json.dumps(asdict(stats), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
