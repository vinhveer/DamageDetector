from __future__ import annotations

import math
from typing import Any

import numpy as np


def area(box: Any) -> float:
    if hasattr(box, "area"):
        return max(0.0, float(getattr(box, "area")))
    return max(0.0, float(getattr(box, "x2") - getattr(box, "x1"))) * max(0.0, float(getattr(box, "y2") - getattr(box, "y1")))


def aspect(box: Any) -> float:
    width = max(1e-6, float(getattr(box, "x2") - getattr(box, "x1")))
    height = max(1e-6, float(getattr(box, "y2") - getattr(box, "y1")))
    return width / height


def box_iou(box_a: Any, box_b: Any) -> float:
    ax1, ay1, ax2, ay2 = (float(getattr(box_a, name)) for name in ("x1", "y1", "x2", "y2"))
    bx1, by1, bx2, by2 = (float(getattr(box_b, name)) for name in ("x1", "y1", "x2", "y2"))
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = area(box_a) + area(box_b) - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def center_distance(box_a: Any, box_b: Any) -> float:
    acx = (float(getattr(box_a, "x1")) + float(getattr(box_a, "x2"))) / 2.0
    acy = (float(getattr(box_a, "y1")) + float(getattr(box_a, "y2"))) / 2.0
    bcx = (float(getattr(box_b, "x1")) + float(getattr(box_b, "x2"))) / 2.0
    bcy = (float(getattr(box_b, "y1")) + float(getattr(box_b, "y2"))) / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    a = np.asarray(emb_a, dtype=np.float32).reshape(-1)
    b = np.asarray(emb_b, dtype=np.float32).reshape(-1)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(float(np.dot(a, b)) / denom, -1.0, 1.0))


def dup_score(box_a: Any, box_b: Any, emb_a: np.ndarray, emb_b: np.ndarray) -> tuple[float, float, float]:
    """Return (dup_score, iou, cos_sim_clamped). Score = IoU × max(0, cos_sim)."""
    iou_value = box_iou(box_a, box_b)
    cos_value = max(0.0, cosine_similarity(emb_a, emb_b))
    return float(iou_value * cos_value), float(iou_value), float(cos_value)


def geometry_score(area_ratio: float) -> float:
    """U-curve favoring small precise boxes (peak in [0.001, 0.30])."""
    ratio = float(area_ratio)
    if ratio <= 0.0:
        return 0.0
    if ratio < 0.001:
        return 0.3
    if ratio <= 0.30:
        return 1.0
    if ratio <= 0.70:
        return 0.5
    return 0.1


# --- C3: containment-aware dedup score + elongation guard --------------------

def elongation(box: Any) -> float:
    """max(aspect_ratio, 1 / aspect_ratio); always >= 1.0."""
    a = aspect(box)
    return max(a, 1.0 / a)


def intersection_area(box_a: Any, box_b: Any) -> float:
    ax1, ay1, ax2, ay2 = (float(getattr(box_a, name)) for name in ("x1", "y1", "x2", "y2"))
    bx1, by1, bx2, by2 = (float(getattr(box_b, name)) for name in ("x1", "y1", "x2", "y2"))
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))


def dup_score_v2(
    box_a: Any, box_b: Any, emb_a: np.ndarray, emb_b: np.ndarray
) -> tuple[float, float, float, float]:
    """Return (score, iou, containment, cos_sim_clamped).

    Spatial component is max(IoU, Containment); score = spatial * clamp(cos, 0, 1).
    Disjoint pairs (intersection <= 0) and missing/zero embeddings score 0.0.
    """
    inter = intersection_area(box_a, box_b)
    if inter <= 0.0:
        return (0.0, 0.0, 0.0, 0.0)
    union = area(box_a) + area(box_b) - inter
    iou_value = inter / max(union, 1e-6)
    containment = inter / max(min(area(box_a), area(box_b)), 1e-6)
    spatial = max(iou_value, containment)
    cos_value = max(0.0, min(1.0, cosine_similarity(emb_a, emb_b)))
    return (float(spatial * cos_value), float(iou_value), float(containment), float(cos_value))
