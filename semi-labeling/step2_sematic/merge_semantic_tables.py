#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def _connect(db_path: Path, *, readonly: bool) -> sqlite3.Connection:
    if readonly:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=60.0)
    else:
        conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _ensure_semantic_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS openclip_semantic_runs (
            semantic_run_id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            source_db_path TEXT NOT NULL,
            source_run_id TEXT NOT NULL,
            source_stage TEXT NOT NULL,
            model_name TEXT NOT NULL,
            pretrained TEXT NOT NULL,
            device TEXT NOT NULL,
            prompt_config_json TEXT NOT NULL,
            options_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS openclip_semantic_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            semantic_run_id TEXT NOT NULL,
            source_detection_id INTEGER NOT NULL,
            source_run_id TEXT NOT NULL,
            image_id INTEGER NOT NULL,
            image_rel_path TEXT NOT NULL,
            image_path TEXT NOT NULL,
            prompt_key TEXT NOT NULL,
            detector_label TEXT NOT NULL,
            detector_score REAL NOT NULL,
            x1 REAL NOT NULL,
            y1 REAL NOT NULL,
            x2 REAL NOT NULL,
            y2 REAL NOT NULL,
            crop_path TEXT NOT NULL,
            status TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            predicted_probability REAL NOT NULL,
            predicted_probability_pct REAL NOT NULL,
            top_prompt TEXT NOT NULL,
            error_type TEXT,
            error_message TEXT,
            raw_json TEXT NOT NULL,
            FOREIGN KEY(semantic_run_id) REFERENCES openclip_semantic_runs(semantic_run_id),
            UNIQUE(semantic_run_id, source_detection_id)
        );

        CREATE TABLE IF NOT EXISTS openclip_semantic_scores (
            result_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            probability REAL NOT NULL,
            probability_pct REAL NOT NULL,
            PRIMARY KEY (result_id, label),
            FOREIGN KEY(result_id) REFERENCES openclip_semantic_results(result_id)
        );

        CREATE INDEX IF NOT EXISTS idx_openclip_semantic_results_detection
        ON openclip_semantic_results(source_detection_id);

        CREATE INDEX IF NOT EXISTS idx_openclip_semantic_results_run
        ON openclip_semantic_results(semantic_run_id, status);
        """
    )
    # C5: carry additive negative-scoring columns through the shard merge.
    existing = {str(row[1]) for row in conn.execute("PRAGMA table_info(openclip_semantic_results)")}
    for name in ("neg_penalty_json", "adjusted_scores_json"):
        if name not in existing:
            conn.execute(f"ALTER TABLE openclip_semantic_results ADD COLUMN {name} TEXT")
    conn.commit()


def _resolve_run_ids(conn: sqlite3.Connection, *, semantic_run_ids: list[str], latest_only: bool) -> list[str]:
    requested = [item.strip() for item in semantic_run_ids if item.strip()]
    if requested:
        return requested
    if latest_only:
        row = conn.execute(
            "SELECT semantic_run_id FROM openclip_semantic_runs ORDER BY created_at_utc DESC LIMIT 1"
        ).fetchone()
        return [str(row["semantic_run_id"])] if row is not None else []
    return [
        str(row["semantic_run_id"])
        for row in conn.execute("SELECT semantic_run_id FROM openclip_semantic_runs ORDER BY created_at_utc, semantic_run_id")
    ]


def _copy_run(target: sqlite3.Connection, row: sqlite3.Row) -> None:
    target.execute(
        """
        INSERT OR IGNORE INTO openclip_semantic_runs (
            semantic_run_id, created_at_utc, source_db_path, source_run_id, source_stage,
            model_name, pretrained, device, prompt_config_json, options_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(row["semantic_run_id"]),
            str(row["created_at_utc"]),
            str(row["source_db_path"]),
            str(row["source_run_id"]),
            str(row["source_stage"]),
            str(row["model_name"]),
            str(row["pretrained"]),
            str(row["device"]),
            str(row["prompt_config_json"]),
            str(row["options_json"]),
        ),
    )


def _opt_col(row: sqlite3.Row, name: str):
    """Return row[name] if the source shard had the column, else None (pre-C5 shard)."""
    return row[name] if name in row.keys() else None


def _copy_result(target: sqlite3.Connection, row: sqlite3.Row) -> int:
    existing = target.execute(
        "SELECT result_id FROM openclip_semantic_results WHERE semantic_run_id = ? AND source_detection_id = ?",
        (str(row["semantic_run_id"]), int(row["source_detection_id"])),
    ).fetchone()
    if existing is not None:
        return int(existing["result_id"])

    cur = target.execute(
        """
        INSERT INTO openclip_semantic_results (
            semantic_run_id, source_detection_id, source_run_id, image_id, image_rel_path, image_path,
            prompt_key, detector_label, detector_score, x1, y1, x2, y2, crop_path, status,
            predicted_label, predicted_probability, predicted_probability_pct, top_prompt,
            error_type, error_message, raw_json, neg_penalty_json, adjusted_scores_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(row["semantic_run_id"]),
            int(row["source_detection_id"]),
            str(row["source_run_id"]),
            int(row["image_id"]),
            str(row["image_rel_path"]),
            str(row["image_path"]),
            str(row["prompt_key"]),
            str(row["detector_label"]),
            float(row["detector_score"]),
            float(row["x1"]),
            float(row["y1"]),
            float(row["x2"]),
            float(row["y2"]),
            str(row["crop_path"]),
            str(row["status"]),
            str(row["predicted_label"]),
            float(row["predicted_probability"]),
            float(row["predicted_probability_pct"]),
            str(row["top_prompt"]),
            None if row["error_type"] is None else str(row["error_type"]),
            None if row["error_message"] is None else str(row["error_message"]),
            str(row["raw_json"]),
            _opt_col(row, "neg_penalty_json"),
            _opt_col(row, "adjusted_scores_json"),
        ),
    )
    return int(cur.lastrowid)


def merge_semantic_tables(
    *,
    target_db: Path,
    source_dbs: list[Path],
    semantic_run_ids: list[str],
    latest_only: bool,
    clear_target: bool,
) -> dict[str, int]:
    target = _connect(target_db, readonly=False)
    try:
        _ensure_semantic_schema(target)
        if clear_target:
            target.execute("DELETE FROM openclip_semantic_scores")
            target.execute("DELETE FROM openclip_semantic_results")
            target.execute("DELETE FROM openclip_semantic_runs")
            target.commit()

        copied_runs = 0
        copied_results = 0
        copied_scores = 0

        for source_db in source_dbs:
            source = _connect(source_db, readonly=True)
            try:
                run_ids = _resolve_run_ids(source, semantic_run_ids=semantic_run_ids, latest_only=latest_only)
                for semantic_run_id in run_ids:
                    run_row = source.execute(
                        "SELECT * FROM openclip_semantic_runs WHERE semantic_run_id = ?",
                        (semantic_run_id,),
                    ).fetchone()
                    if run_row is None:
                        continue
                    _copy_run(target, run_row)
                    copied_runs += 1

                    result_rows = source.execute(
                        "SELECT * FROM openclip_semantic_results WHERE semantic_run_id = ? ORDER BY result_id",
                        (semantic_run_id,),
                    ).fetchall()
                    for result_row in result_rows:
                        target_result_id = _copy_result(target, result_row)
                        copied_results += 1
                        score_rows = source.execute(
                            "SELECT * FROM openclip_semantic_scores WHERE result_id = ? ORDER BY label",
                            (int(result_row["result_id"]),),
                        ).fetchall()
                        for score_row in score_rows:
                            target.execute(
                                """
                                INSERT OR REPLACE INTO openclip_semantic_scores (result_id, label, probability, probability_pct)
                                VALUES (?, ?, ?, ?)
                                """,
                                (
                                    int(target_result_id),
                                    str(score_row["label"]),
                                    float(score_row["probability"]),
                                    float(score_row["probability_pct"]),
                                ),
                            )
                            copied_scores += 1
                    target.commit()
            finally:
                source.close()

        return {
            "runs": copied_runs,
            "results": copied_results,
            "scores": copied_scores,
        }
    finally:
        target.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge OpenCLIP semantic tables from shard DBs into one final DB.")
    parser.add_argument("--target-db", required=True, help="Destination SQLite DB that will receive merged semantic tables.")
    parser.add_argument("--source-db", action="append", required=True, help="Source shard SQLite DB; repeat for each shard.")
    parser.add_argument("--semantic-run-id", action="append", default=[], help="Specific semantic_run_id to merge; repeat as needed.")
    parser.add_argument("--latest-only", action="store_true", help="Merge only the latest semantic run from each source DB.")
    parser.add_argument("--clear-target", action="store_true", help="Delete existing semantic rows in target DB before merging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = merge_semantic_tables(
        target_db=Path(args.target_db).expanduser().resolve(),
        source_dbs=[Path(item).expanduser().resolve() for item in list(args.source_db or [])],
        semantic_run_ids=[str(item) for item in list(args.semantic_run_id or [])],
        latest_only=bool(args.latest_only),
        clear_target=bool(args.clear_target),
    )
    print(f"target_db={Path(args.target_db).expanduser().resolve()}")
    print(f"merged_runs={summary['runs']}")
    print(f"merged_results={summary['results']}")
    print(f"merged_scores={summary['scores']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
