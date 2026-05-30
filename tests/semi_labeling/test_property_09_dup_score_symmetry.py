# Feature: semi-labeling-pipeline-improvements, Property 9: Dup-score symmetry
# For all detection pairs (A, B), Dup_Score(A, B) == Dup_Score(B, A).
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
def test_dup_score_symmetry(a, b, ea, eb):
    ab = dup_score_v2(_mk(a), _mk(b), ea, eb)
    ba = dup_score_v2(_mk(b), _mk(a), eb, ea)
    assert ab == ba
