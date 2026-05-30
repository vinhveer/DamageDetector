"""C2A: cross-class containment flagging (non-breaking post-filter).

Annotates (never drops) kept detections that are strongly contained inside a kept
detection of a *different* predicted label and are visually similar, so a reviewer can
inspect likely cross-class mistakes. Only ``drop_reason`` is changed; ``keep`` is never
touched (Requirement 2.1-2.3).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any

from output_store import DedupDecision
from pair_features import area, cosine_similarity, intersection_area

SUSPECT_REASON = "cross_class_containment_suspect"


def _containment(det_a: Any, det_b: Any) -> float:
    return intersection_area(det_a, det_b) / max(min(area(det_a), area(det_b)), 1e-6)


def _suspect_result_id(det_a: Any, det_b: Any) -> int:
    """The smaller-area detection; on equal area, the larger result_id."""
    area_a, area_b = area(det_a), area(det_b)
    if area_a < area_b:
        return int(det_a.result_id)
    if area_b < area_a:
        return int(det_b.result_id)
    return max(int(det_a.result_id), int(det_b.result_id))


def apply_cross_class_containment(
    decisions: list[DedupDecision],
    detections_by_id: dict[int, Any],
    embedding_map: dict[int, Any],
    *,
    containment_threshold: float = 0.85,
    cos_sim_threshold: float = 0.80,
) -> list[DedupDecision]:
    kept_by_image: dict[str, list[DedupDecision]] = defaultdict(list)
    for decision in decisions:
        if decision.keep:
            kept_by_image[decision.image_rel_path].append(decision)

    flagged: dict[int, str] = {}
    for kept in kept_by_image.values():
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                a, b = kept[i], kept[j]
                if str(a.predicted_label) == str(b.predicted_label):
                    continue
                det_a = detections_by_id.get(int(a.result_id))
                det_b = detections_by_id.get(int(b.result_id))
                if det_a is None or det_b is None:
                    continue
                if _containment(det_a, det_b) < containment_threshold:
                    continue
                emb_a = embedding_map.get(int(a.result_id))
                emb_b = embedding_map.get(int(b.result_id))
                if emb_a is None or emb_b is None:
                    continue
                if cosine_similarity(emb_a, emb_b) < cos_sim_threshold:
                    continue
                flagged[_suspect_result_id(det_a, det_b)] = SUSPECT_REASON

    if not flagged:
        return list(decisions)
    return [
        replace(decision, drop_reason=flagged[int(decision.result_id)])
        if decision.keep and int(decision.result_id) in flagged
        else decision
        for decision in decisions
    ]
