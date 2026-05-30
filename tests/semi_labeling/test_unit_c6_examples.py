# C6 example/edge unit tests (Requirements 6.2, 6.9, 6.10).
from __future__ import annotations

import os
import sqlite3
import tempfile

import pytest

import sl_imports

_store = sl_imports.load_step_file("step7_label_review", "corrections/store.py", "sl_corr_store")
_cal = sl_imports.load_step_file("step7_label_review", "corrections/calibration.py", "sl_calibration")
LabelCorrection = _store.LabelCorrection
CorrectionStore = _store.CorrectionStore
CorrectionWriteError = _store.CorrectionWriteError


def _correction(correction_id, correction_type):
    return LabelCorrection(
        correction_id=correction_id, created_at_utc="t", source_run_id="r", result_id=1,
        image_rel_path="i.png", original_label="crack", corrected_label="mold",
        correction_type=correction_type,
    )


def test_correction_type_validation():  # Req 6.2
    with tempfile.TemporaryDirectory() as d:
        store = CorrectionStore(sqlite3.connect(os.path.join(d, "c.db")))
        with pytest.raises(ValueError):
            store.add_correction(_correction("c1", "bogus"))
        # valid types accepted
        for ctype in ("relabel", "confirm", "reject"):
            store.add_correction(_correction(f"ok_{ctype}", ctype))


def test_transactional_write_failure_leaves_prior_rows():  # Req 6.9
    with tempfile.TemporaryDirectory() as d:
        store = CorrectionStore(sqlite3.connect(os.path.join(d, "c.db")))
        store.add_correction(_correction("dup", "confirm"))
        # Duplicate primary key -> write fails, rolls back, raises CorrectionWriteError(id).
        with pytest.raises(CorrectionWriteError) as exc_info:
            store.add_correction(_correction("dup", "relabel"))
        assert exc_info.value.correction_id == "dup"
        rows = list(store.iter_corrections())
        assert len(rows) == 1 and rows[0].correction_type == "confirm"  # prior row intact


def test_calibration_fallback_to_base():  # Req 6.10
    # Unreadable / missing file -> {} -> base threshold applied.
    missing = _cal.load_calibration("/nonexistent/path/calibration.json")
    assert missing == {}
    assert _cal.effective_threshold(missing, "crack", base=0.40) == 0.40
    # Present calibration but label absent -> base threshold.
    calib = {"label_thresholds": {"crack": {"base": 0.40, "delta": 0.05, "effective": 0.45}}}
    assert _cal.effective_threshold(calib, "mold", base=0.33) == 0.33
    assert _cal.effective_threshold(calib, "crack") == 0.45
