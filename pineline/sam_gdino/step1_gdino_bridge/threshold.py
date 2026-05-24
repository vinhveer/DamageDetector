from __future__ import annotations

from typing import Iterable


def apply_dynamic_threshold(
    detections: list[dict],
    *,
    score_floor: float,
    top_k: int,
) -> tuple[list[dict], float]:
    """Filter detections by floor + top-k per image.

    Returns (kept_detections, best_score). If best_score < score_floor,
    the image is considered as not containing a bridge and all detections
    are dropped.
    """
    if not detections:
        return [], 0.0
    scored = []
    for det in detections:
        try:
            score = float(det.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        scored.append((score, det))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score = scored[0][0]
    if best_score < float(score_floor):
        return [], best_score
    above = [(s, d) for s, d in scored if s >= float(score_floor)]
    kept = [d for _, d in above[: max(1, int(top_k))]]
    return kept, best_score


def relabel(detections: Iterable[dict], label: str = "bridge") -> list[dict]:
    out: list[dict] = []
    for det in detections:
        det = dict(det)
        det["label"] = label
        out.append(det)
    return out
