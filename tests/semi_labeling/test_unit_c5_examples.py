# C5 example/edge unit tests (Requirements 5.5, 5.7).
from __future__ import annotations

import os
import sqlite3
import tempfile

import sl_imports

_neg = sl_imports.load_step("step2_sematic", "negatives")
_merge = sl_imports.load_step("step2_sematic", "merge_semantic_tables")


def test_all_zero_renormalization_guard():  # Req 5.5
    # sum(adjusted) <= 0 -> leave scores at 0.0 with no division.
    out = _neg.renormalize({"crack": 0.0, "mold": 0.0, "spall": 0.0})
    assert out == {"crack": 0.0, "mold": 0.0, "spall": 0.0}
    # alpha large enough to drive every adjusted score to 0.
    adjusted = _neg.adjust_scores({"crack": 0.5}, {"crack": 1.0}, {"crack": 10.0})
    assert adjusted == {"crack": 0.0}
    assert _neg.renormalize(adjusted) == {"crack": 0.0}


_RUN_INSERT = (
    "INSERT INTO openclip_semantic_runs (semantic_run_id, created_at_utc, source_db_path, source_run_id, "
    "source_stage, model_name, pretrained, device, prompt_config_json, options_json) "
    "VALUES ('r', 't', 'p', 'sr', 'final', 'm', 'pt', 'cpu', '{}', '{}')"
)
# Legacy (pre-C5) results schema: no neg columns.
_LEGACY_RESULTS = """
CREATE TABLE openclip_semantic_runs (
    semantic_run_id TEXT PRIMARY KEY, created_at_utc TEXT, source_db_path TEXT, source_run_id TEXT,
    source_stage TEXT, model_name TEXT, pretrained TEXT, device TEXT, prompt_config_json TEXT, options_json TEXT
);
CREATE TABLE openclip_semantic_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT, semantic_run_id TEXT, source_detection_id INTEGER,
    source_run_id TEXT, image_id INTEGER, image_rel_path TEXT, image_path TEXT, prompt_key TEXT,
    detector_label TEXT, detector_score REAL, x1 REAL, y1 REAL, x2 REAL, y2 REAL, crop_path TEXT, status TEXT,
    predicted_label TEXT, predicted_probability REAL, predicted_probability_pct REAL, top_prompt TEXT,
    error_type TEXT, error_message TEXT, raw_json TEXT
);
CREATE TABLE openclip_semantic_scores (
    result_id INTEGER, label TEXT, probability REAL, probability_pct REAL, PRIMARY KEY (result_id, label)
);
"""

_RESULT_COLS = (
    "semantic_run_id, source_detection_id, source_run_id, image_id, image_rel_path, image_path, prompt_key, "
    "detector_label, detector_score, x1, y1, x2, y2, crop_path, status, predicted_label, predicted_probability, "
    "predicted_probability_pct, top_prompt, error_type, error_message, raw_json"
)
_RESULT_VALS = "'r', 1, 'sr', 1, 'i.png', 'i.png', 'crack', 'damage', 0.9, 0, 0, 10, 10, '', 'ok', 'crack', 0.9, 90.0, 'tp', NULL, NULL, '{}'"


def _make_source(path, *, c5):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    if c5:
        _merge._ensure_semantic_schema(conn)  # includes neg columns
        conn.execute(_RUN_INSERT)
        conn.execute(
            f"INSERT INTO openclip_semantic_results ({_RESULT_COLS}, neg_penalty_json, adjusted_scores_json) "
            f"VALUES ({_RESULT_VALS}, '{{\"crack\": 0.2}}', '{{\"crack\": 0.8}}')"
        )
    else:
        conn.executescript(_LEGACY_RESULTS)
        conn.execute(_RUN_INSERT)
        conn.execute(f"INSERT INTO openclip_semantic_results ({_RESULT_COLS}) VALUES ({_RESULT_VALS})")
    conn.commit()
    conn.close()


def test_merge_carries_negative_columns():  # Req 5.7 + sharding/merge
    with tempfile.TemporaryDirectory() as d:
        c5_db = os.path.join(d, "shard_c5.sqlite3")
        legacy_db = os.path.join(d, "shard_legacy.sqlite3")
        target = os.path.join(d, "merged.sqlite3")
        _make_source(c5_db, c5=True)

        from pathlib import Path

        _merge.merge_semantic_tables(
            target_db=Path(target), source_dbs=[Path(c5_db)],
            semantic_run_ids=[], latest_only=False, clear_target=False,
        )
        conn = sqlite3.connect(target)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT neg_penalty_json, adjusted_scores_json, raw_json FROM openclip_semantic_results").fetchone()
        assert row["neg_penalty_json"] == '{"crack": 0.2}'
        assert row["adjusted_scores_json"] == '{"crack": 0.8}'
        assert row["raw_json"] == "{}"  # raw positive scores preserved
        conn.close()

        # Pre-C5 shard merges with NULL neg columns (backward compatible).
        _make_source(legacy_db, c5=False)
        target2 = os.path.join(d, "merged2.sqlite3")
        _merge.merge_semantic_tables(
            target_db=Path(target2), source_dbs=[Path(legacy_db)],
            semantic_run_ids=[], latest_only=False, clear_target=False,
        )
        conn2 = sqlite3.connect(target2)
        conn2.row_factory = sqlite3.Row
        row2 = conn2.execute("SELECT neg_penalty_json, adjusted_scores_json FROM openclip_semantic_results").fetchone()
        assert row2["neg_penalty_json"] is None and row2["adjusted_scores_json"] is None
        conn2.close()
