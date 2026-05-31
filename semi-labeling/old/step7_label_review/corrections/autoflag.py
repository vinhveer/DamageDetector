"""C6: flag new detections similar to known correction errors. stdlib + numpy."""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def find_similar_to_corrections(
    new_embeddings: np.ndarray,
    new_ids: list[int],
    correction_embeddings: np.ndarray,
    correction_labels: list[str],
    *,
    similarity_threshold: float = 0.92,
) -> dict[int, str]:
    """{result_id: 'similar_to_known_error:<original_label>'} when max cosine similarity
    to any stored correction embedding >= threshold.

    When ``correction_embeddings`` is empty (no correction carries a crop embedding, i.e.
    C4 has not produced any), this logs an explicit warning and returns {} (a visible
    no-op, never a silent clean pass)."""
    corr = np.asarray(correction_embeddings, dtype=np.float32)
    if corr.ndim != 2 or corr.shape[0] == 0 or corr.shape[1] == 0:
        logger.warning("autoflag skipped: no correction embeddings (requires C4 multi-view crop embeddings)")
        return {}

    new = np.asarray(new_embeddings, dtype=np.float32)
    if new.ndim != 2 or new.shape[0] == 0:
        return {}

    new_n = _normalize_rows(new)
    corr_n = _normalize_rows(corr)
    sims = new_n @ corr_n.T  # (num_new, num_corr), cosine in [-1, 1]

    flags: dict[int, str] = {}
    for row_index, result_id in enumerate(new_ids):
        best = int(np.argmax(sims[row_index]))
        if float(sims[row_index, best]) >= float(similarity_threshold):
            flags[int(result_id)] = f"similar_to_known_error:{correction_labels[best]}"
    return flags
