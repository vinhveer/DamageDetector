from __future__ import annotations

from typing import Sequence

import numpy as np


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def dedup_within_image(
    detections: list[dict],
    embeddings: np.ndarray,
    *,
    iou_threshold: float,
    cosine_threshold: float,
) -> list[dict]:
    """Decide kept/dropped for each detection. Greedy: sort by clip_prob desc,
    drop any later detection whose IoU and cosine both exceed thresholds with
    an already-kept one.

    Returns list of decision dicts: parent_image_id, det_idx, kept(0/1),
    merged_into_det_idx (None if kept), reason.
    """
    if not detections:
        return []
    if embeddings.ndim != 2 or embeddings.shape[0] != len(detections):
        raise ValueError("embeddings shape mismatch with detections")
    # rank by CLIP prob desc, tiebreak by GDINO score desc
    order = sorted(
        range(len(detections)),
        key=lambda i: (-float(detections[i].get("clip_prob") or 0.0),
                       -float(detections[i].get("gdino_score") or 0.0)),
    )

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    normed = embeddings / norms

    kept: list[int] = []
    decisions = [None] * len(detections)  # type: ignore[list-item]

    for i in order:
        det_i = detections[i]
        merged_into = None
        for j in kept:
            if detections[j]["parent_image_id"] != det_i["parent_image_id"]:
                continue
            iou_ij = iou(det_i["box"], detections[j]["box"])
            if iou_ij < float(iou_threshold):
                continue
            cos = float(np.dot(normed[i], normed[j]))
            if cos < float(cosine_threshold):
                continue
            merged_into = detections[j]["det_idx"]
            break
        if merged_into is None:
            kept.append(i)
            decisions[i] = {
                "parent_image_id": det_i["parent_image_id"],
                "det_idx": det_i["det_idx"],
                "kept": 1,
                "merged_into_det_idx": None,
                "reason": "kept",
            }
        else:
            decisions[i] = {
                "parent_image_id": det_i["parent_image_id"],
                "det_idx": det_i["det_idx"],
                "kept": 0,
                "merged_into_det_idx": int(merged_into),
                "reason": (
                    f"iou>={iou_threshold:.2f}, cos>={cosine_threshold:.2f}"
                ),
            }
    return decisions  # type: ignore[return-value]
