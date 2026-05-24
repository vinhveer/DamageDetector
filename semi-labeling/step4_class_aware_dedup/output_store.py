from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class DedupDecision:
    result_id: int
    image_rel_path: str
    predicted_label: str
    keep: bool
    fused: bool
    duplicate_group_id: str
    representative_id: int
    p_dup_max: float
    p_good: float
    drop_reason: str
    fused_bbox: list[float] | None = None


def connect_output(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS dedup_runs (
            dedup_run_id              TEXT PRIMARY KEY,
            created_at_utc            TEXT NOT NULL,
            source_db_path            TEXT NOT NULL,
            embedding_db_path         TEXT NOT NULL,
            embedding_run_id          TEXT NOT NULL,
            duplicate_classifier_json TEXT NOT NULL,
            quality_classifier_json   TEXT NOT NULL,
            options_json              TEXT NOT NULL,
            total_detections          INTEGER NOT NULL,
            kept_count                INTEGER NOT NULL,
            fused_count               INTEGER NOT NULL,
            dropped_count             INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS dedup_results (
            dedup_run_id        TEXT NOT NULL,
            result_id           INTEGER NOT NULL,
            image_rel_path      TEXT NOT NULL,
            predicted_label     TEXT NOT NULL,
            keep                INTEGER NOT NULL,
            fused               INTEGER NOT NULL,
            duplicate_group_id  TEXT NOT NULL,
            representative_id   INTEGER NOT NULL,
            p_dup_max           REAL NOT NULL,
            p_good              REAL NOT NULL,
            drop_reason         TEXT NOT NULL,
            fused_bbox_json     TEXT,
            PRIMARY KEY (dedup_run_id, result_id)
        );

        CREATE INDEX IF NOT EXISTS idx_dedup_image
        ON dedup_results (dedup_run_id, image_rel_path, keep);

        CREATE INDEX IF NOT EXISTS idx_dedup_group
        ON dedup_results (dedup_run_id, duplicate_group_id);

        CREATE TABLE IF NOT EXISTS dedup_pair_scores (
            dedup_run_id    TEXT NOT NULL,
            result_id_a     INTEGER NOT NULL,
            result_id_b     INTEGER NOT NULL,
            p_dup           REAL NOT NULL,
            features_json   TEXT NOT NULL,
            PRIMARY KEY (dedup_run_id, result_id_a, result_id_b)
        );

        CREATE INDEX IF NOT EXISTS idx_dedup_pairs_score
        ON dedup_pair_scores (dedup_run_id, p_dup);
        """
    )
    conn.commit()


def insert_run_metadata(
    conn: sqlite3.Connection,
    *,
    dedup_run_id: str,
    source_db_path: Path,
    embedding_db_path: Path,
    embedding_run_id: str,
    duplicate_classifier_json: str,
    quality_classifier_json: str,
    options: dict[str, Any],
    total_detections: int,
) -> None:
    conn.execute(
        """
        INSERT INTO dedup_runs (
            dedup_run_id, created_at_utc, source_db_path, embedding_db_path, embedding_run_id,
            duplicate_classifier_json, quality_classifier_json, options_json,
            total_detections, kept_count, fused_count, dropped_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0)
        """,
        (
            dedup_run_id,
            utc_now(),
            str(Path(source_db_path).expanduser().resolve()),
            str(Path(embedding_db_path).expanduser().resolve()),
            str(embedding_run_id),
            duplicate_classifier_json,
            quality_classifier_json,
            json.dumps(options, ensure_ascii=False, sort_keys=True),
            int(total_detections),
        ),
    )
    conn.commit()


def persist_pair_scores(
    conn: sqlite3.Connection,
    *,
    dedup_run_id: str,
    pair_keys: list[tuple[int, int]],
    p_dups: Iterable[float],
    pair_features: Iterable[dict[str, float]],
) -> int:
    rows = [
        (
            dedup_run_id,
            int(result_id_a),
            int(result_id_b),
            float(p_dup),
            json.dumps(features, ensure_ascii=False, sort_keys=True),
        )
        for (result_id_a, result_id_b), p_dup, features in zip(pair_keys, p_dups, pair_features, strict=True)
    ]
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO dedup_pair_scores (
            dedup_run_id, result_id_a, result_id_b, p_dup, features_json
        ) VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def write_decisions(conn: sqlite3.Connection, *, dedup_run_id: str, decisions: Iterable[DedupDecision]) -> int:
    rows = []
    for decision in decisions:
        fused_bbox_json = None if decision.fused_bbox is None else json.dumps([float(v) for v in decision.fused_bbox], ensure_ascii=False)
        rows.append(
            (
                dedup_run_id,
                int(decision.result_id),
                decision.image_rel_path,
                decision.predicted_label,
                1 if decision.keep else 0,
                1 if decision.fused else 0,
                decision.duplicate_group_id,
                int(decision.representative_id),
                float(decision.p_dup_max),
                float(decision.p_good),
                decision.drop_reason,
                fused_bbox_json,
            )
        )
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO dedup_results (
            dedup_run_id, result_id, image_rel_path, predicted_label, keep, fused,
            duplicate_group_id, representative_id, p_dup_max, p_good, drop_reason, fused_bbox_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def finalize_run_counts(conn: sqlite3.Connection, *, dedup_run_id: str) -> tuple[int, int, int]:
    kept = int(
        conn.execute(
            "SELECT COUNT(*) FROM dedup_results WHERE dedup_run_id = ? AND keep = 1 AND fused = 0",
            (dedup_run_id,),
        ).fetchone()[0]
    )
    fused = int(
        conn.execute(
            "SELECT COUNT(*) FROM dedup_results WHERE dedup_run_id = ? AND keep = 1 AND fused = 1",
            (dedup_run_id,),
        ).fetchone()[0]
    )
    dropped = int(
        conn.execute(
            "SELECT COUNT(*) FROM dedup_results WHERE dedup_run_id = ? AND keep = 0",
            (dedup_run_id,),
        ).fetchone()[0]
    )
    conn.execute(
        "UPDATE dedup_runs SET kept_count = ?, fused_count = ?, dropped_count = ? WHERE dedup_run_id = ?",
        (kept, fused, dropped, dedup_run_id),
    )
    conn.commit()
    return kept, fused, dropped


def latest_dedup_run_id(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT dedup_run_id FROM dedup_runs ORDER BY created_at_utc DESC, dedup_run_id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("No dedup run found.")
    return str(row["dedup_run_id"])
