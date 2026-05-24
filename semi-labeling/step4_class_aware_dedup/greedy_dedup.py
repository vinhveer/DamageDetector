from __future__ import annotations

import uuid
from typing import Any

import numpy as np

from output_store import DedupDecision
from pair_features import area, dup_score, geometry_score
from source_store import Detection


def _p_good(detection: Detection) -> float:
    semantic = max(0.0, min(1.0, float(detection.predicted_probability_pct) / 100.0))
    detector = max(0.0, min(1.0, float(detection.detector_score)))
    geom = geometry_score(float(detection.area_ratio))
    return float(0.5 * semantic + 0.3 * detector + 0.2 * geom)


def _group_id(image_rel_path: str, anchor_result_id: int) -> str:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"{image_rel_path}|anchor={anchor_result_id}").hex


def greedy_dedup(
    image_detections: list[Detection],
    image_embeddings: np.ndarray,
    *,
    dup_threshold: float,
    pair_logger: Any | None = None,
) -> list[DedupDecision]:
    if not image_detections:
        return []

    order = sorted(
        range(len(image_detections)),
        key=lambda i: (area(image_detections[i]), int(image_detections[i].result_id)),
        reverse=True,
    )

    kept_indices: list[int] = []
    kept_group_ids: dict[int, str] = {}
    decisions_by_id: dict[int, DedupDecision] = {}
    p_dup_max_by_id: dict[int, float] = {int(d.result_id): 0.0 for d in image_detections}

    threshold = float(dup_threshold)

    for idx_b in order:
        box_b = image_detections[idx_b]
        emb_b = image_embeddings[idx_b]
        label_b = str(box_b.predicted_label)
        winner_idx: int | None = None
        best_score: float = 0.0

        for idx_k in kept_indices:
            box_k = image_detections[idx_k]
            if str(box_k.predicted_label) != label_b:
                continue
            score, iou_value, cos_value = dup_score(box_b, box_k, emb_b, image_embeddings[idx_k])
            p_dup_max_by_id[int(box_b.result_id)] = max(p_dup_max_by_id[int(box_b.result_id)], score)
            p_dup_max_by_id[int(box_k.result_id)] = max(p_dup_max_by_id[int(box_k.result_id)], score)
            if pair_logger is not None:
                pair_logger(int(box_k.result_id), int(box_b.result_id), score, iou_value, cos_value)
            if score >= threshold and score > best_score:
                winner_idx = idx_k
                best_score = score

        if winner_idx is None:
            group_id = _group_id(box_b.image_rel_path, int(box_b.result_id))
            kept_indices.append(idx_b)
            kept_group_ids[idx_b] = group_id
            decisions_by_id[int(box_b.result_id)] = DedupDecision(
                result_id=int(box_b.result_id),
                image_rel_path=box_b.image_rel_path,
                predicted_label=label_b,
                keep=True,
                fused=False,
                duplicate_group_id=group_id,
                representative_id=int(box_b.result_id),
                p_dup_max=0.0,
                p_good=_p_good(box_b),
                drop_reason="",
            )
        else:
            winner = image_detections[winner_idx]
            decisions_by_id[int(box_b.result_id)] = DedupDecision(
                result_id=int(box_b.result_id),
                image_rel_path=box_b.image_rel_path,
                predicted_label=label_b,
                keep=False,
                fused=False,
                duplicate_group_id=kept_group_ids[winner_idx],
                representative_id=int(winner.result_id),
                p_dup_max=0.0,
                p_good=_p_good(box_b),
                drop_reason="duplicate",
            )

    final: list[DedupDecision] = []
    for detection in image_detections:
        rid = int(detection.result_id)
        decision = decisions_by_id[rid]
        final.append(
            DedupDecision(
                result_id=decision.result_id,
                image_rel_path=decision.image_rel_path,
                predicted_label=decision.predicted_label,
                keep=decision.keep,
                fused=decision.fused,
                duplicate_group_id=decision.duplicate_group_id,
                representative_id=decision.representative_id,
                p_dup_max=float(p_dup_max_by_id.get(rid, 0.0)),
                p_good=decision.p_good,
                drop_reason=decision.drop_reason,
            )
        )
    return final
