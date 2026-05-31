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


# --- C4: multi-view schema, encoding, resume, and reads ----------------------

import numpy as np  # noqa: E402

_EMB_DTYPE = "<f4"


def encode_vector(arr: Any) -> bytes:
    return np.asarray(arr, dtype=_EMB_DTYPE).tobytes()


def decode_vector(blob: bytes, dim: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=_EMB_DTYPE)
    if arr.size != int(dim):
        raise ValueError(f"Invalid embedding size: got {arr.size}, expected {dim}")
    return arr.astype(np.float32, copy=True)


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")]


def ensure_view_schema(conn: sqlite3.Connection) -> None:
    """Migrate detection_embeddings / skipped_detections to a per-view primary key.

    Idempotent. The legacy PK (embedding_run_id, result_id) is rebuilt to
    (embedding_run_id, result_id, view_name) via a transactional table copy that
    defaults existing rows to the 'tight' view and verifies row-count parity before
    dropping the old table (Requirement 4.7; Error Handling -> C4 PK rebuild).
    """
    ensure_schema(conn)
    if "view_name" not in _columns(conn, "detection_embeddings"):
        _rebuild_with_view(
            conn,
            table="detection_embeddings",
            create_sql="""
                CREATE TABLE detection_embeddings_v2 (
                    embedding_run_id  TEXT NOT NULL,
                    result_id         INTEGER NOT NULL,
                    image_rel_path    TEXT NOT NULL,
                    predicted_label   TEXT NOT NULL,
                    view_name         TEXT NOT NULL DEFAULT 'tight',
                    embedding_blob    BLOB NOT NULL,
                    PRIMARY KEY (embedding_run_id, result_id, view_name)
                )
            """,
            copy_sql="""
                INSERT INTO detection_embeddings_v2
                    (embedding_run_id, result_id, image_rel_path, predicted_label, view_name, embedding_blob)
                SELECT embedding_run_id, result_id, image_rel_path, predicted_label, 'tight', embedding_blob
                FROM detection_embeddings
            """,
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_emb_view ON detection_embeddings(embedding_run_id, view_name, result_id)"
        )
        conn.commit()
    if "view_name" not in _columns(conn, "skipped_detections"):
        _rebuild_with_view(
            conn,
            table="skipped_detections",
            create_sql="""
                CREATE TABLE skipped_detections_v2 (
                    embedding_run_id  TEXT NOT NULL,
                    result_id         INTEGER NOT NULL,
                    view_name         TEXT NOT NULL DEFAULT 'tight',
                    reason            TEXT NOT NULL,
                    PRIMARY KEY (embedding_run_id, result_id, view_name)
                )
            """,
            copy_sql="""
                INSERT INTO skipped_detections_v2 (embedding_run_id, result_id, view_name, reason)
                SELECT embedding_run_id, result_id, 'tight', reason FROM skipped_detections
            """,
        )
        conn.commit()


def _rebuild_with_view(conn: sqlite3.Connection, *, table: str, create_sql: str, copy_sql: str) -> None:
    old_count = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    conn.execute(f"DROP TABLE IF EXISTS {table}_v2")
    conn.execute(create_sql)
    conn.execute(copy_sql)
    new_count = int(conn.execute(f"SELECT COUNT(*) FROM {table}_v2").fetchone()[0])
    if new_count != old_count:
        conn.execute(f"DROP TABLE IF EXISTS {table}_v2")
        conn.commit()
        raise RuntimeError(f"{table} migration row-count mismatch: {old_count} -> {new_count}")
    conn.execute(f"DROP TABLE {table}")
    conn.execute(f"ALTER TABLE {table}_v2 RENAME TO {table}")
    conn.commit()


def bulk_insert_view_embeddings(
    conn: sqlite3.Connection, rows: Iterable[tuple[str, int, str, str, str, bytes]]
) -> int:
    """rows: (embedding_run_id, result_id, image_rel_path, predicted_label, view_name, blob)."""
    rows = list(rows)
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO detection_embeddings
            (embedding_run_id, result_id, image_rel_path, predicted_label, view_name, embedding_blob)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def bulk_insert_view_skipped(conn: sqlite3.Connection, rows: Iterable[tuple[str, int, str, str]]) -> int:
    """rows: (embedding_run_id, result_id, view_name, reason)."""
    rows = list(rows)
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO skipped_detections (embedding_run_id, result_id, view_name, reason)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def embedded_view_keys(conn: sqlite3.Connection, *, embedding_run_id: str) -> set[tuple[int, str]]:
    rows = conn.execute(
        "SELECT result_id, COALESCE(view_name, 'tight') AS view_name FROM detection_embeddings WHERE embedding_run_id = ?",
        (embedding_run_id,),
    ).fetchall()
    return {(int(row["result_id"]), str(row["view_name"])) for row in rows}


def skipped_view_keys(conn: sqlite3.Connection, *, embedding_run_id: str) -> set[tuple[int, str]]:
    rows = conn.execute(
        "SELECT result_id, COALESCE(view_name, 'tight') AS view_name FROM skipped_detections WHERE embedding_run_id = ?",
        (embedding_run_id,),
    ).fetchall()
    return {(int(row["result_id"]), str(row["view_name"])) for row in rows}


def load_tight_embeddings(
    conn: sqlite3.Connection, *, embedding_run_id: str, dim: int, result_ids: Iterable[int]
) -> tuple[dict[int, np.ndarray], set[int]]:
    """Return ({result_id: tight_vector}, missing_ids) for the requested ids; NULL
    view_name rows count as 'tight' (Requirement 4.6, 4.7)."""
    requested = [int(item) for item in result_ids]
    out: dict[int, np.ndarray] = {}
    if not requested:
        return out, set()
    batch_size = 900
    for start in range(0, len(requested), batch_size):
        chunk = requested[start : start + batch_size]
        placeholders = ", ".join("?" for _ in chunk)
        rows = conn.execute(
            f"""
            SELECT result_id, embedding_blob FROM detection_embeddings
            WHERE embedding_run_id = ? AND COALESCE(view_name, 'tight') = 'tight'
              AND result_id IN ({placeholders})
            """,
            [embedding_run_id, *chunk],
        ).fetchall()
        for row in rows:
            out[int(row["result_id"])] = decode_vector(row["embedding_blob"], dim)
    missing = {rid for rid in requested if rid not in out}
    return out, missing
