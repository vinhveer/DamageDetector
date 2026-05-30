# Feature: semi-labeling-pipeline-improvements, Property 23: Calibration math bounds and monotonicity
# For all label stats with total_reviewed>0, error_rate in [0,1], total_reviewed==0 labels
# excluded, delta in [0,0.15] and monotonically non-decreasing in error_rate.
from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

import sl_imports

_cal = sl_imports.load_step_file("step7_label_review", "corrections/calibration.py", "sl_calibration")
LabelCalibrationStats = _cal.LabelCalibrationStats


@st.composite
def _stat(draw, label="crack"):
    confirmed = draw(st.integers(0, 100))
    relabeled = draw(st.integers(0, 100))
    rejected = draw(st.integers(0, 100))
    total = confirmed + relabeled + rejected
    return LabelCalibrationStats(label, total, confirmed, relabeled, rejected)


@given(stat=_stat(), stat0=_stat())
def test_calibration_bounds_monotonic(stat, stat0):
    if stat.total_reviewed > 0:
        assert 0.0 <= stat.error_rate <= 1.0
        assert 0.0 <= stat.suggested_threshold_delta <= 0.15

    # Excludes total_reviewed == 0 labels.
    zero = LabelCalibrationStats("mold", 0, 0, 0, 0)
    calib = _cal.compute_calibration([stat0, zero])
    assert "mold" not in calib["label_thresholds"]


@given(er1=st.floats(0.0, 1.0), er2=st.floats(0.0, 1.0))
def test_delta_monotonic_in_error_rate(er1, er2):
    lo, hi = sorted([er1, er2])

    def delta(er):
        # Build a stat whose error_rate == er with total_reviewed=1000.
        bad = round(er * 1000)
        return LabelCalibrationStats("crack", 1000, 1000 - bad, bad, 0).suggested_threshold_delta

    assert delta(hi) >= delta(lo) - 1e-12
