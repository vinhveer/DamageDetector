# Feature: semi-labeling-pipeline-improvements, Property 19: Penalty bounds and monotonic adjustment
# negative penalty in [0,1]; pre-renormalization Adjusted_Score in [0, positive_score];
# increasing the penalty (fixed pos & alpha) never increases the Adjusted_Score.
from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

import sl_imports
from sl_strategies import score_dicts

_neg = sl_imports.load_step("step2_sematic", "negatives")
_unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
_alpha = st.floats(min_value=0.0, max_value=10.0, allow_nan=False)


@given(
    neg_sims=st.lists(st.floats(-1.0, 1.0, allow_nan=False), min_size=1, max_size=6),
    pos=score_dicts(),
    alpha_val=_alpha,
    p1=_unit, p2=_unit,
)
def test_penalty_bounds_and_monotonic(neg_sims, pos, alpha_val, p1, p2):
    penalty = _neg.negative_penalty(np.asarray(neg_sims, dtype=np.float32))
    assert 0.0 <= penalty <= 1.0

    labels = list(pos.keys())
    alpha = {label: alpha_val for label in labels}

    # Pre-renormalization adjusted score is in [0, positive_score].
    adj = _neg.adjust_scores(pos, {label: penalty for label in labels}, alpha)
    for label in labels:
        assert 0.0 <= adj[label] <= pos[label] + 1e-9

    # Monotonic: a larger penalty never yields a larger adjusted score (fixed pos, alpha).
    lo, hi = (p1, p2) if p1 <= p2 else (p2, p1)
    a_lo = _neg.adjust_scores(pos, {label: lo for label in labels}, alpha)
    a_hi = _neg.adjust_scores(pos, {label: hi for label in labels}, alpha)
    for label in labels:
        assert a_hi[label] <= a_lo[label] + 1e-9
