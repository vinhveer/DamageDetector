# Feature: semi-labeling-pipeline-improvements, Property 28: Correction round-trip
# Persisting then reading back a Label_Correction preserves all recorded fields and keeps
# confidence in [0, 1].
from __future__ import annotations

import os
import sqlite3
import tempfile

from hypothesis import given
from hypothesis import strategies as st

import sl_imports

_store = sl_imports.load_step_file("step7_label_review", "corrections/store.py", "sl_corr_store")
LabelCorrection = _store.LabelCorrection
CorrectionStore = _store.CorrectionStore

_text = st.text(min_size=0, max_size=12)


@given(
    correction_id=st.text(min_size=1, max_size=16).filter(lambda s: s.strip()),
    source_run_id=_text,
    result_id=st.integers(0, 10_000),
    image_rel_path=_text,
    original_label=st.sampled_from(["crack", "mold", "spall"]),
    corrected_label=st.sampled_from(["crack", "mold", "spall", "reject"]),
    correction_type=st.sampled_from(["relabel", "confirm", "reject"]),
    confidence=st.floats(-2.0, 3.0, allow_nan=False),
)
def test_correction_round_trip(
    correction_id, source_run_id, result_id, image_rel_path,
    original_label, corrected_label, correction_type, confidence,
):
    correction = LabelCorrection(
        correction_id=correction_id, created_at_utc="2026-05-30T00:00:00+00:00",
        source_run_id=source_run_id, result_id=result_id, image_rel_path=image_rel_path,
        original_label=original_label, corrected_label=corrected_label,
        correction_type=correction_type, confidence=confidence,
    )
    assert 0.0 <= correction.confidence <= 1.0  # clamped at construction

    with tempfile.TemporaryDirectory() as d:
        conn = sqlite3.connect(os.path.join(d, "c.db"))
        store = CorrectionStore(conn)
        store.add_correction(correction)
        rows = list(store.iter_corrections())
        assert len(rows) == 1
        got = rows[0]
        assert got.correction_id == correction.correction_id
        assert got.result_id == correction.result_id
        assert got.original_label == correction.original_label
        assert got.corrected_label == correction.corrected_label
        assert got.correction_type == correction.correction_type
        assert abs(got.confidence - correction.confidence) <= 1e-9
        assert 0.0 <= got.confidence <= 1.0
        conn.close()
