# Feature: semi-labeling-pipeline-improvements, Property 21: Zero-alpha identity
# When alpha is 0.0 for every label, the renormalized Adjusted_Scores equal the
# renormalized positive scores.
from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

import sl_imports
from sl_strategies import score_dicts

_neg = sl_imports.load_step("step2_sematic", "negatives")


@given(pos=score_dicts(), penalties=score_dicts())
def test_zero_alpha_identity(pos, penalties):
    labels = list(pos.keys())
    alpha = {label: 0.0 for label in labels}
    adjusted = _neg.adjust_scores(pos, penalties, alpha)
    out = _neg.renormalize(adjusted)
    expected = _neg.renormalize(pos)
    for label in labels:
        assert abs(out[label] - expected[label]) <= 1e-9
