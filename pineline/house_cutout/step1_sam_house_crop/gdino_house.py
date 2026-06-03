from __future__ import annotations

from pathlib import Path
from typing import Callable

from pineline.house_cutout.step1_sam_house_crop.prompts import role_for_label


def _box_to_xyxy(box) -> tuple[float, float, float, float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def detect_house_and_negatives(
    service,
    work_path: Path,
    *,
    width: int,
    height: int,
    checkpoint: str,
    text_queries: list[str],
    box_threshold: float,
    text_threshold: float,
    max_dets: int,
    device: str,
    tiled_threshold: int,
    score_floor: float,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Chạy GDINO trên working RGB, trả về (house_boxes, neg_boxes).

    - house_boxes: detection vai trò 'house' có score >= score_floor (đã sort giảm dần).
    - neg_boxes:   detection vai trò 'negative' (window/door), score >= score_floor.
    Mỗi phần tử là dict {"box": (x1,y1,x2,y2), "score": float, "label": str}.
    """
    params = {
        "gdino_checkpoint": checkpoint,
        "gdino_config_id": "auto",
        "text_queries": list(text_queries),
        "box_threshold": float(box_threshold),
        "text_threshold": float(text_threshold),
        "max_dets": int(max_dets),
        "device": device,
    }

    max_dim = max(int(width), int(height))
    if max_dim > int(tiled_threshold):
        result = service.call(
            "recursive_detect",
            {
                "image_path": str(work_path),
                "params": params,
                "target_labels": list(text_queries),
                "max_depth": 1,
                "min_box_px": 32,
            },
            log_fn=log_fn,
        )
    else:
        result = service.call(
            "predict",
            {"image_path": str(work_path), "params": params},
            log_fn=log_fn,
        )

    detections = list(result.get("detections") or [])

    house: list[dict] = []
    negatives: list[dict] = []
    for det in detections:
        xyxy = _box_to_xyxy(det.get("box"))
        if xyxy is None:
            continue
        score = float(det.get("score") or 0.0)
        if score < float(score_floor):
            continue
        label = str(det.get("label") or "")
        entry = {"box": xyxy, "score": score, "label": label}
        if role_for_label(label) == "negative":
            negatives.append(entry)
        else:
            house.append(entry)

    house.sort(key=lambda d: d["score"], reverse=True)
    negatives.sort(key=lambda d: d["score"], reverse=True)
    return house, negatives
