# Feature: semi-labeling-pipeline-improvements, Property 6: Reorder coverage
# For all step1 detection sets, the set embedded under the full-reorder path equals the
# entire detection set -- no label-based filtering removes any detection before embedding.
from __future__ import annotations

import os
import sqlite3
import tempfile

from hypothesis import given
from hypothesis import strategies as st

import sl_imports

read_detections = sl_imports.load_step("step3_embedding", "embed_detections").read_detections

_SCHEMA = """
CREATE TABLE runs (run_id TEXT, input_dir TEXT);
CREATE TABLE images (image_id INTEGER, width INTEGER, height INTEGER);
CREATE TABLE openclip_semantic_results (
    result_id INTEGER, semantic_run_id TEXT, source_run_id TEXT, image_id INTEGER,
    image_rel_path TEXT, image_path TEXT, predicted_label TEXT,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL, status TEXT, predicted_probability_pct REAL
);
"""


@given(labels=st.lists(st.sampled_from(["crack", "mold", "spall"]), min_size=1, max_size=8))
def test_reorder_coverage(labels):
    with tempfile.TemporaryDirectory() as d:
        conn = sqlite3.connect(os.path.join(d, "s.db"))
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        conn.execute("INSERT INTO runs VALUES ('run', '/imgs')")
        conn.execute("INSERT INTO images VALUES (1, 100, 100)")
        for rid, label in enumerate(labels):
            conn.execute(
                "INSERT INTO openclip_semantic_results VALUES (?, 'sr', 'run', 1, 'i.png', 'i.png', ?, 0, 0, 10, 10, 'ok', 90.0)",
                (rid, label),
            )
        conn.commit()

        all_ids = {rid for rid, _ in enumerate(labels)}
        # Reorder path: no label filter -> every detection is embedded.
        embedded = {d.result_id for d in read_detections(conn, semantic_run_id="sr", min_confidence_pct=0.0, labels=[], limit=0)}
        assert embedded == all_ids

        # A label filter is a strict subset, proving coverage came from dropping the filter.
        only_crack = {d.result_id for d in read_detections(conn, semantic_run_id="sr", min_confidence_pct=0.0, labels=["crack"], limit=0)}
        assert only_crack == {rid for rid, label in enumerate(labels) if label == "crack"}
        assert only_crack <= embedded
        conn.close()
