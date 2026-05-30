# Feature: semi-labeling-pipeline-improvements, Property 7: Dup-score bounds
# For all detection pairs and embeddings, Dup_Score in [0,1] with Cos_Sim clamped to
# [0,1] before multiplication.
from __future__ import annotations

from types import SimpleNamespace

from hypothesis import given

import sl_imports
from sl_strategies import boxes, embedding_vectors

dup_score_v2 = sl_imports.load_step("step4_class_aware_dedup", "pair_features").dup_score_v2


def _mk(t):
    x1, y1, x2, y2 = t
    return SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2)


@given(a=boxes(), b=boxes(), ea=embedding_vectors(), eb=embedding_vectors())
def test_dup_score_bounds(a, b, ea, eb):
    score, iou, cont, cos = dup_score_v2(_mk(a), _mk(b), ea, eb)
    assert 0.0 <= score <= 1.0
    assert 0.0 <= iou <= 1.0
    assert 0.0 <= cont <= 1.0
    assert 0.0 <= cos <= 1.0
    assert abs(score - max(iou, cont) * cos) <= 1e-9
