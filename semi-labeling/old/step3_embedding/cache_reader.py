from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).expanduser().resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=60.0)
    conn.row_factory = sqlite3.Row
    return conn


def _latest_run(conn: sqlite3.Connection, *, model_name: str) -> sqlite3.Row:
    row = conn.execute(
        """
        SELECT embedding_run_id, dim
        FROM embedding_runs
        WHERE model_name = ?
        ORDER BY created_at_utc DESC, embedding_run_id DESC
        LIMIT 1
        """,
        (model_name,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"No embedding run found for model_name={model_name}")
    return row


def _decode_embedding(blob: bytes, *, dim: int, result_id: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype="<f4")
    if arr.size != int(dim):
        raise ValueError(f"Invalid embedding size for result_id={result_id}: got {arr.size}, expected {dim}")
    return arr.astype(np.float32, copy=True)


def load_embeddings(
    db_path: Path,
    *,
    model_name: str = "facebook/dinov2-giant",
    result_ids: list[int] | None = None,
) -> tuple[np.ndarray, list[int]]:
    """
    Returns (embeddings, result_ids).

    embeddings is a float32 array of shape (N, dim). If result_ids is provided,
    returned rows follow that requested order for IDs found in the latest run.
    """
    conn = _connect_readonly(Path(db_path))
    try:
        run = _latest_run(conn, model_name=model_name)
        embedding_run_id = str(run["embedding_run_id"])
        dim = int(run["dim"])
        if result_ids is None:
            rows = conn.execute(
                """
                SELECT result_id, embedding_blob
                FROM detection_embeddings
                WHERE embedding_run_id = ?
                ORDER BY result_id
                """,
                (embedding_run_id,),
            ).fetchall()
        else:
            requested = [int(item) for item in result_ids]
            if not requested:
                return np.empty((0, dim), dtype=np.float32), []
            placeholders = ", ".join("?" for _ in requested)
            rows = conn.execute(
                f"""
                SELECT result_id, embedding_blob
                FROM detection_embeddings
                WHERE embedding_run_id = ? AND result_id IN ({placeholders})
                """,
                [embedding_run_id, *requested],
            ).fetchall()
            by_id = {int(row["result_id"]): row for row in rows}
            rows = [by_id[item] for item in requested if item in by_id]
    finally:
        conn.close()

    ids = [int(row["result_id"]) for row in rows]
    vectors = [_decode_embedding(row["embedding_blob"], dim=dim, result_id=int(row["result_id"])) for row in rows]
    if not vectors:
        return np.empty((0, dim), dtype=np.float32), []
    return np.vstack(vectors).astype(np.float32, copy=False), ids
