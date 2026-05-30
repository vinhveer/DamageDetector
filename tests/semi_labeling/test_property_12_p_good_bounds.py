# Feature: semi-labeling-pipeline-improvements, Property 12: P_Good bounds and input sanitization
# For all detections (incl. missing/non-numeric semantic_prob, detector_score, geometry
# inputs), P_Good in [0,1] and equals 0.5*s + 0.3*d + 0.2*geometry_score, each term
# coerced to 0.0 when missing/non-numeric and clamped to [0,1].
from __future__ import annotations

from types import SimpleNamespace

from hypothesis import given
from hypothesis import strategies as st

import sl_imports

_gd = sl_imports.load_step("step4_class_aware_dedup", "greedy_dedup")
_num01 = _gd._num01
_p_good = _gd._p_good
geometry_score = sl_imports.load_step("step4_class_aware_dedup", "pair_features").geometry_score

# Messy inputs: None, NaN/inf, strings, and well-formed values.
_messy = st.one_of(
    st.none(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(max_size=4),
    st.floats(min_value=-2.0, max_value=2.0),
)


def _coerce_ratio(r):
    try:
        v = float(r)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if v != v else v


@given(pct=_messy, detector=_messy, ratio=_messy)
def test_p_good_bounds_and_sanitization(pct, detector, ratio):
    d = SimpleNamespace(predicted_probability_pct=pct, detector_score=detector, area_ratio=ratio)
    pg = _p_good(d)
    assert 0.0 <= pg <= 1.0
    expected = (
        0.5 * _num01(pct, 1.0 / 100.0)
        + 0.3 * _num01(detector)
        + 0.2 * geometry_score(_coerce_ratio(ratio))
    )
    assert abs(pg - expected) <= 1e-9
