from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from torch_runtime import get_torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def overlay_mask(bgr: np.ndarray, mask01: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    out = bgr.copy()
    mask = mask01.astype(bool)
    out[mask] = (alpha * color + (1 - alpha) * out[mask]).astype(np.uint8)
    return out


def filter_small_components(mask01: np.ndarray, min_area: int) -> np.ndarray:
    if int(min_area) <= 0:
        return mask01
    mask = (mask01 > 0).astype(np.uint8)
    if int(np.count_nonzero(mask)) == 0:
        return mask
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    keep = np.zeros(num, dtype=np.uint8)
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            keep[idx] = 1
    return keep[labels].astype(np.uint8)


def _mask_bbox(mask01: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _mask_stats(mask01: np.ndarray, image_shape: tuple[int, int]) -> dict[str, float] | None:
    mask = (mask01 > 0).astype(np.uint8)
    area = int(np.count_nonzero(mask))
    if area <= 0:
        return None
    bbox = _mask_bbox(mask)
    if bbox is None:
        return None
    h_img, w_img = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = bbox
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    bbox_area = max(1, bbox_w * bbox_h)
    eps = 1e-6
    edges = cv2.Canny(mask * 255, 50, 150)
    perimeter = float(np.count_nonzero(edges))
    return {
        "area": float(area),
        "fill_ratio": float(area) / max(float(bbox_area), eps),
        "image_ratio": float(area) / max(float(max(1, h_img * w_img)), eps),
        "elongation": float(max(bbox_w, bbox_h)) / max(float(max(1, min(bbox_w, bbox_h))), eps),
        "thinness": perimeter / max(float(np.sqrt(float(area) + 1.0)), eps),
        "bbox_w": float(bbox_w),
        "bbox_h": float(bbox_h),
    }


def score_mask_darkness(mask01: np.ndarray, gray: np.ndarray) -> float:
    mask = (mask01 > 0).astype(np.uint8)
    if int(np.count_nonzero(mask)) == 0:
        return float("-inf")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    ring = ((dilated > 0) & (mask == 0))
    inside = mask.astype(bool)
    if not np.any(ring) or not np.any(inside):
        return 0.0
    inside_mean = float(np.mean(gray[inside]))
    ring_mean = float(np.mean(gray[ring]))
    return ring_mean - inside_mean


def score_mask_for_crack(mask01: np.ndarray, sam_score: float, image_shape: tuple[int, int]) -> float:
    mask = (mask01 > 0).astype(np.uint8)
    area = int(np.count_nonzero(mask))
    if area <= 0:
        return float("-inf")
    h_img, w_img = int(image_shape[0]), int(image_shape[1])
    bbox = _mask_bbox(mask)
    if bbox is None:
        return float("-inf")
    x1, y1, x2, y2 = bbox
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    bbox_area = max(1, bbox_w * bbox_h)
    eps = 1e-6
    fill_ratio = float(area) / max(float(bbox_area), eps)
    image_ratio = float(area) / max(float(max(1, h_img * w_img)), eps)
    elongation = float(max(bbox_w, bbox_h)) / max(float(max(1, min(bbox_w, bbox_h))), eps)
    edges = cv2.Canny(mask * 255, 50, 150)
    perimeter = float(np.count_nonzero(edges))
    thinness = perimeter / max(float(np.sqrt(float(area) + 1.0)), eps)
    score = 0.0
    score += 1.5 * min(elongation, 12.0)
    score += 2.0 * thinness
    score += 1.5 * (1.0 - min(fill_ratio, 1.0))
    score += 0.75 * float(sam_score)
    if image_ratio > 0.35:
        score -= 8.0
    elif image_ratio > 0.2:
        score -= 4.0
    return float(score)


def select_sam_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    image_shape: tuple[int, int],
    *,
    prefer_crack: bool,
) -> np.ndarray:
    if masks is None or len(masks) == 0:
        raise ValueError("SAM returned no masks.")
    if not prefer_crack:
        return masks[int(np.argmax(scores))].astype(np.uint8)
    fallback_idx = int(np.argmax(scores))
    best_idx = 0
    best_val = float("-inf")
    for idx, mask in enumerate(masks):
        try:
            value = score_mask_for_crack(mask.astype(np.uint8), float(scores[idx]), image_shape)
        except Exception:
            value = float("-inf")
        if value > best_val:
            best_val = value
            best_idx = idx
    if not np.isfinite(best_val):
        return masks[fallback_idx].astype(np.uint8)
    return masks[int(best_idx)].astype(np.uint8)


def make_sam_auto_mask_generator(sam_model: Any, profile: str = "ULTRA") -> SamAutomaticMaskGenerator:
    upper = str(profile or "ULTRA").strip().upper()
    if upper == "FAST":
        generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=16,
            points_per_batch=64,
            pred_iou_thresh=0.9,
            stability_score_thresh=0.9,
            stability_score_offset=0.95,
            box_nms_thresh=0.7,
            crop_n_layers=0,
            crop_overlap_ratio=0.45,
            crop_nms_thresh=0.7,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=120,
            output_mode="binary_mask",
        )
    elif upper == "QUALITY":
        generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=24,
            points_per_batch=64,
            pred_iou_thresh=0.92,
            stability_score_thresh=0.92,
            stability_score_offset=0.95,
            box_nms_thresh=0.7,
            crop_n_layers=1,
            crop_overlap_ratio=0.5,
            crop_nms_thresh=0.7,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=80,
            output_mode="binary_mask",
        )
    else:
        generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=32,
            points_per_batch=128,
            pred_iou_thresh=0.94,
            stability_score_thresh=0.94,
            stability_score_offset=0.95,
            box_nms_thresh=0.7,
            crop_n_layers=2,
            crop_overlap_ratio=0.55,
            crop_nms_thresh=0.7,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=40,
            output_mode="binary_mask",
        )

    generator.point_grids = [np.asarray(grid, dtype=np.float32) for grid in generator.point_grids]
    original_apply_coords = generator.predictor.transform.apply_coords
    original_apply_boxes = generator.predictor.transform.apply_boxes

    def _apply_coords_float32(coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        transformed = original_apply_coords(coords, original_size)
        return np.asarray(transformed, dtype=np.float32)

    def _apply_boxes_float32(boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        transformed = original_apply_boxes(boxes, original_size)
        return np.asarray(transformed, dtype=np.float32)

    generator.predictor.transform.apply_coords = _apply_coords_float32
    generator.predictor.transform.apply_boxes = _apply_boxes_float32
    return generator


def load_checkpoint_state_dict(checkpoint_path: str) -> dict:
    torch = get_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)} ({checkpoint_path})")


def infer_sam_model_type_from_state_dict(state_dict: dict) -> Optional[str]:
    dim = None
    if "image_encoder.pos_embed" in state_dict:
        tensor = state_dict["image_encoder.pos_embed"]
        try:
            dim = int(tensor.shape[-1])
        except Exception:
            dim = None
    elif "image_encoder.patch_embed.proj.weight" in state_dict:
        tensor = state_dict["image_encoder.patch_embed.proj.weight"]
        try:
            dim = int(tensor.shape[0])
        except Exception:
            dim = None
    if dim == 768:
        return "vit_b"
    if dim == 1024:
        return "vit_l"
    if dim == 1280:
        return "vit_h"
    return None


def load_sam_model(checkpoint_path: str, requested_model_type: str) -> Tuple[Any, str]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")
    state_dict = load_checkpoint_state_dict(checkpoint_path)
    inferred = infer_sam_model_type_from_state_dict(state_dict)
    requested = (requested_model_type or "auto").strip().lower()
    if requested == "auto":
        if inferred is None:
            raise RuntimeError(
                "Cannot infer SAM model type from checkpoint. "
                "Choose the correct one explicitly: vit_b, vit_l, or vit_h."
            )
        model_type = inferred
    else:
        model_type = inferred or requested
    if model_type not in sam_model_registry:
        raise ValueError(f"Unknown SAM model type: {model_type!r}")
    sam_model = sam_model_registry[model_type](checkpoint=None)
    try:
        sam_model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"SAM checkpoint/model type mismatch. requested={requested_model_type!r}, inferred={inferred!r}.\n{exc}"
        ) from exc
    return sam_model, model_type
