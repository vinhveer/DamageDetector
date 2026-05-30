# Feature: semi-labeling-pipeline-improvements, Property 24: Calibration round-trip
# Writing calibration to file and reading it back yields the same effective thresholds
# (base + delta) within floating-point tolerance.
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

import sl_imports

_cal = sl_imports.load_step_file("step7_label_review", "corrections/calibration.py", "sl_calibration")


@st.composite
def _stats(draw):
    out = []
    for label in ("crack", "mold", "spall"):
        confirmed = draw(st.integers(0, 200))
        relabeled = draw(st.integers(0, 200))
        rejected = draw(st.integers(0, 200))
        out.append(_cal.LabelCalibrationStats(label, confirmed + relabeled + rejected, confirmed, relabeled, rejected))
    return out


@given(stats=_stats(), bases=st.dictionaries(st.sampled_from(["crack", "mold", "spall"]), st.floats(0.1, 0.9)))
def test_calibration_round_trip(stats, bases):
    calib = _cal.compute_calibration(stats, base_thresholds=bases)
    with tempfile.TemporaryDirectory() as d:
        path = Path(os.path.join(d, "calibration.json"))
        _cal.write_calibration(path, calib)
        loaded = _cal.load_calibration(path)

    for label in calib["label_thresholds"]:
        before = _cal.effective_threshold(calib, label)
        after = _cal.effective_threshold(loaded, label)
        assert abs(before - after) <= 1e-9
        # effective == base + delta
        entry = calib["label_thresholds"][label]
        assert abs(after - (entry["base"] + entry["delta"])) <= 1e-9
