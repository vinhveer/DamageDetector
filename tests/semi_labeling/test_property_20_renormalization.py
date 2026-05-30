# Feature: semi-labeling-pipeline-improvements, Property 20: Renormalization sums to one
# For all score sets whose pre-normalization sum is > 0, the renormalized Adjusted_Scores
# sum to 1.0 within 1e-6.
from __future__ import annotations

from hypothesis import assume, given

import sl_imports
from sl_strategies import score_dicts

renormalize = sl_imports.load_step("step2_sematic", "negatives").renormalize


@given(adjusted=score_dicts())
def test_renormalization_sums_to_one(adjusted):
    assume(sum(adjusted.values()) > 0.0)
    out = renormalize(adjusted)
    assert abs(sum(out.values()) - 1.0) <= 1e-6
