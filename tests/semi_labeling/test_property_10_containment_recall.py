# Feature: semi-labeling-pipeline-improvements, Property 10: Containment recall
# For all pairs where one box is fully contained in another (Containment = 1.0) and
# Cos_Sim >= 0.10, Dup_Score >= 0.10, so nested duplicates are detected.
from __future__ import annotations

from types import SimpleNamespace

from hypothesis import assume, given
from hypothesis import strategies as st

import sl_imports
from sl_strategies import embedding_vectors

dup_score_v2 = sl_imports.load_step("step4_class_aware_dedup", "pair_features").dup_score_v2

_dim = st.floats(min_value=10.0, max_value=200.0, allow_nan=False)
_frac = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


@given(
    bw=_dim, bh=_dim,
    fx=_frac, fy=_frac, fw=st.floats(0.05, 1.0), fh=st.floats(0.05, 1.0),
    ea=embedding_vectors(), eb=embedding_vectors(),
)
def test_containment_recall(bw, bh, fx, fy, fw, fh, ea, eb):
    big = SimpleNamespace(x1=0.0, y1=0.0, x2=bw, y2=bh)
    # A small box fully inside big => containment(small, big) == 1.0.
    sx1 = fx * bw * 0.5
    sy1 = fy * bh * 0.5
    sx2 = sx1 + max(0.5, fw * (bw - sx1))
    sy2 = sy1 + max(0.5, fh * (bh - sy1))
    small = SimpleNamespace(x1=sx1, y1=sy1, x2=min(sx2, bw), y2=min(sy2, bh))

    score, iou, cont, cos = dup_score_v2(small, big, ea, eb)
    assume(cos >= 0.10)
    assert cont >= 1.0 - 1e-9
    assert score >= 0.10 - 1e-9
