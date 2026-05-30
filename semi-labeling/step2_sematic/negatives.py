"""C5: negative-anchor scoring (pure core).

All functions operate on plain numbers / numpy arrays so the correctness properties run
without a model. ``adjusted_scores_for_image`` is the reusable path the classifier calls
with its cached negative-anchor text features.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np


def _normalize(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0.0 else arr


def cosine_to_texts(image_feat, text_feats) -> np.ndarray:
    """Cosine similarity in [-1, 1] between an image feature and each text feature row,
    returned in row order (the cosine layer of ``similarity_to_texts``)."""
    img = _normalize(image_feat)
    mat = np.asarray(text_feats, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    sims = [float(np.clip(float(np.dot(img, _normalize(row))), -1.0, 1.0)) for row in mat]
    return np.asarray(sims, dtype=np.float32)


def negative_penalty(neg_sims) -> float:
    """Max negative-anchor similarity, clamped to [0, 1]; 0.0 when there are no anchors."""
    arr = np.asarray(neg_sims, dtype=np.float64).reshape(-1)
    return float(np.clip(arr.max(initial=0.0), 0.0, 1.0))


def adjust_scores(
    pos_scores: Mapping[str, float],
    neg_penalty: Mapping[str, float],
    alpha: Mapping[str, float],
) -> dict[str, float]:
    """adjusted[l] = max(0, pos[l] - alpha[l] * neg[l]); no renormalization here."""
    return {
        label: max(0.0, float(pos) - float(alpha.get(label, 0.0)) * float(neg_penalty.get(label, 0.0)))
        for label, pos in pos_scores.items()
    }


def renormalize(adjusted: Mapping[str, float]) -> dict[str, float]:
    """Divide by the sum when it is > 0 (sums to 1 within 1e-6); otherwise unchanged."""
    total = float(sum(float(v) for v in adjusted.values()))
    if total > 0.0:
        return {label: float(v) / total for label, v in adjusted.items()}
    return {label: float(v) for label, v in adjusted.items()}


def adjusted_scores_for_image(
    image_feat,
    pos_scores: Mapping[str, float],
    neg_text_feats_by_label: Mapping[str, np.ndarray],
    alpha: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (per-label negative penalty, renormalized adjusted scores) for one image."""
    penalties: dict[str, float] = {}
    for label in pos_scores:
        feats = neg_text_feats_by_label.get(label)
        if feats is None or len(feats) == 0:
            penalties[label] = 0.0
        else:
            penalties[label] = negative_penalty(cosine_to_texts(image_feat, feats))
    return penalties, renormalize(adjust_scores(pos_scores, penalties, alpha))
