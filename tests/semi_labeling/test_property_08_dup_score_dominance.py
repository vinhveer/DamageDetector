# Feature: semi-labeling-pipeline-improvements, Property 8: Dominance over the legacy formula
# For all detection pairs, max(IoU, Containment) * Cos_Sim >= IoU * Cos_Sim, so the new
# Dup_Score is never smaller than the legacy IoU-based score.
from __future__ import annotations

from types import SimpleNamespace

from hypothesis import given
from hypothesis import strategies as st

import sl_imports
from sl_strategies import embedding_vectors

_pf = sl_imports.load_step("step4_class_aware_dedup", "pair_features")
dup_score = _pf.dup_score
dup_score_v2 = _pf.dup_score_v2

_pos = st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False)
_side = st.floats(min_value=0.1, max_value=200.0, allow_nan=False, allow_infinity=False)


def _box(x1, y1, w, h):
    return SimpleNamespace(x1=x1, y1=y1, x2=x1 + w, y2=y1 + h)


@given(
    ax=_pos, ay=_pos, aw=_side, ah=_side,
    bx=_pos, by=_pos, bw=_side, bh=_side,
    ea=embedding_vectors(), eb=embedding_vectors(),
)
def test_dominance_over_legacy(ax, ay, aw, ah, bx, by, bw, bh, ea, eb):
    ba, bb = _box(ax, ay, aw, ah), _box(bx, by, bw, bh)
    s2, iou2, cont2, cos2 = dup_score_v2(ba, bb, ea, eb)
    s1, _iou1, _cos1 = dup_score(ba, bb, ea, eb)

    assert s2 >= iou2 * cos2 - 1e-12      # spatial dominance (pure inequality)
    assert s2 >= s1 - 1e-9                # never smaller than the legacy score
