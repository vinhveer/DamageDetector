from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def connect_output(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).expanduser().resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=60.0)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS embedding_runs (
            embedding_run_id        TEXT PRIMARY KEY,
            created_at_utc          TEXT NOT NULL,
            source_db_path          TEXT NOT NULL,
            source_semantic_run_id  TEXT NOT NULL,
            model_name              TEXT NOT NULL,
            dim                     INTEGER NOT NULL,
            device                  TEXT NOT NULL,
            padding_ratio           REAL NOT NULL,
            total_detections        INTEGER NOT NULL,
            embedded_count          INTEGER NOT NULL,
            skipped_count           INTEGER NOT NULL,
            options_json            TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS detection_embeddings (
            embedding_run_id  TEXT NOT NULL,
            result_id         INTEGER NOT NULL,
            image_rel_path    TEXT NOT NULL,
            predicted_label   TEXT NOT NULL,
            embedding_blob    BLOB NOT NULL,
            PRIMARY KEY (embedding_run_id, result_id)
        );

        CREATE INDEX IF NOT EXISTS idx_embeddings_image
        ON detection_embeddings (embedding_run_id, image_rel_path);

        CREATE INDEX IF NOT EXISTS idx_embeddings_label
        ON detection_embeddings (embedding_run_id, predicted_label);

        CREATE TABLE IF NOT EXISTS skipped_detections (
            embedding_run_id  TEXT NOT NULL,
            result_id         INTEGER NOT NULL,
            reason            TEXT NOT NULL,
            PRIMARY KEY (embedding_run_id, result_id)
        );
        """
    )
    conn.commit()


def latest_matching_run(conn: sqlite3.Connection, *, model_name: str, semantic_run_id: str) -> sqlite3.Row | None:
    ensure_schema(conn)
    return conn.execute(
        """
        SELECT * FROM embedding_runs
        WHERE model_name = ? AND source_semantic_run_id = ?
        ORDER BY created_at_utc DESC, embedding_run_id DESC
        LIMIT 1
        """,
        (model_name, semantic_run_id),
    ).fetchone()


def insert_run_metadata(
    conn: sqlite3.Connection,
    *,
    embedding_run_id: str,
    source_db_path: Path,
    source_semantic_run_id: str,
    model_name: str,
    dim: int,
    device: str,
    padding_ratio: float,
    total_detections: int,
    options: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO embedding_runs (
            embedding_run_id, created_at_utc, source_db_path, source_semantic_run_id,
            model_name, dim, device, padding_ratio, total_detections,
            embedded_count, skipped_count, options_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?)
        """,
        (
            embedding_run_id,
            utc_now(),
            str(Path(source_db_path).expanduser().resolve()),
            source_semantic_run_id,
            model_name,
            int(dim),
            device,
            float(padding_ratio),
            int(total_detections),
            json.dumps(options, ensure_ascii=False, sort_keys=True),
        ),
    )
    conn.commit()


def embedded_result_ids(conn: sqlite3.Connection, *, embedding_run_id: str) -> set[int]:
    rows = conn.execute(
        "SELECT result_id FROM detection_embeddings WHERE embedding_run_id = ?",
        (embedding_run_id,),
    ).fetchall()
    return {int(row["result_id"]) for row in rows}


def skipped_result_ids(conn: sqlite3.Connection, *, embedding_run_id: str) -> set[int]:
    rows = conn.execute(
        "SELECT result_id FROM skipped_detections WHERE embedding_run_id = ?",
        (embedding_run_id,),
    ).fetchall()
    return {int(row["result_id"]) for row in rows}


def bulk_insert_embeddings(conn: sqlite3.Connection, rows: Iterable[tuple[str, int, str, str, bytes]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO detection_embeddings (
            embedding_run_id, result_id, image_rel_path, predicted_label, embedding_blob
        ) VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def bulk_insert_skipped(conn: sqlite3.Connection, rows: Iterable[tuple[str, int, str]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO skipped_detections (embedding_run_id, result_id, reason)
        VALUES (?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def update_run_counts(conn: sqlite3.Connection, *, embedding_run_id: str) -> tuple[int, int]:
    embedded = int(
        conn.execute(
            "SELECT COUNT(*) FROM detection_embeddings WHERE embedding_run_id = ?",
            (embedding_run_id,),
        ).fetchone()[0]
    )
    skipped = int(
        conn.execute(
            "SELECT COUNT(*) FROM skipped_detections WHERE embedding_run_id = ?",
            (embedding_run_id,),
        ).fetchone()[0]
    )
    conn.execute(
        "UPDATE embedding_runs SET embedded_count = ?, skipped_count = ? WHERE embedding_run_id = ?",
        (embedded, skipped, embedding_run_id),
    )
    conn.commit()
    return embedded, skipped
