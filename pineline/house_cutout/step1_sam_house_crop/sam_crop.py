from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def predict_mask_with_points(
    predictor: Any,
    positive_points: list[tuple[float, float]],
    negative_points: list[tuple[float, float]],
    *,
    multimask_output: bool = True,
) -> np.ndarray:
    """Chạy SAM với điểm dương (label=1) và điểm âm (label=0) → mask bool tốt nhất.

    Đây là điểm khác chính so với pipeline sam_gdino: window/door được đưa vào như
    điểm âm để mask nhà không nuốt khung cửa / cửa sổ.
    """
    if not positive_points:
        return np.zeros((0, 0), dtype=bool)

    pos = list(positive_points)
    neg = list(negative_points or [])
    coords = np.array(pos + neg, dtype=np.float32)
    labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)

    masks, scores, _ = predictor.predict(
        point_coords=coords,
        point_labels=labels,
        multimask_output=bool(multimask_output),
    )
    if multimask_output:
        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(bool)
    return masks[0].astype(bool)


def union_masks(masks: list[np.ndarray]) -> np.ndarray:
    masks = [m for m in masks if m.size > 0]
    if not masks:
        return np.zeros((0, 0), dtype=bool)
    out = masks[0].copy()
    for m in masks[1:]:
        if m.shape != out.shape:
            continue
        np.logical_or(out, m, out=out)
    return out


def mask_bbox(mask: np.ndarray, *, pad_px: int = 0) -> tuple[int, int, int, int] | None:
    if mask.size == 0 or not mask.any():
        return None
    ys, xs = np.where(mask)
    h, w = mask.shape[:2]
    x1 = max(0, int(xs.min()) - pad_px)
    y1 = max(0, int(ys.min()) - pad_px)
    x2 = min(w, int(xs.max()) + 1 + pad_px)
    y2 = min(h, int(ys.max()) + 1 + pad_px)
    return x1, y1, x2, y2


def write_cutout_and_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    cutout_path: Path,
    mask_path: Path,
) -> int:
    """Ghi cutout RGBA (nền trong suốt) cắt theo bbox của mask + mask nhị phân.

    Pixel ngoài mask có alpha=0, trong mask giữ RGB gốc với alpha=255. Đây chính
    là định dạng cutout mà step2 (dino_cutout-style) mong đợi.
    """
    x1, y1, x2, y2 = bbox
    rgb_crop = image_rgb[y1:y2, x1:x2]
    mask_crop = (mask[y1:y2, x1:x2] > 0).astype(np.uint8)

    rgba = np.zeros((rgb_crop.shape[0], rgb_crop.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = rgb_crop
    rgba[..., 3] = mask_crop * 255

    cutout_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    # cv2 ghi BGRA: convert RGB->BGR, giữ alpha.
    bgra = np.dstack([cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR), rgba[..., 3]])
    cv2.imwrite(str(cutout_path), bgra)
    cv2.imwrite(str(mask_path), mask_crop * 255)
    return int(mask.sum())
