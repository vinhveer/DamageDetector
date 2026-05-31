from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class KeptBox:
    result_id: int
    image_rel_path: str
    predicted_label: str


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).expanduser().resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def resolve_dedup_run_id(conn: sqlite3.Connection, requested: str) -> str:
    raw = str(requested or "latest").strip()
    if raw and raw.lower() != "latest":
        row = conn.execute(
            "SELECT dedup_run_id FROM dedup_runs WHERE dedup_run_id = ?",
            (raw,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Dedup run not found: {raw}")
        return raw
    row = conn.execute(
        "SELECT dedup_run_id FROM dedup_runs ORDER BY created_at_utc DESC, dedup_run_id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("No dedup run found in dedup DB.")
    return str(row["dedup_run_id"])


def read_kept_boxes(
    conn: sqlite3.Connection,
    *,
    dedup_run_id: str,
    labels: Iterable[str] | None = None,
) -> list[KeptBox]:
    label_list = [str(item).strip() for item in (labels or []) if str(item).strip()]
    clauses = ["dedup_run_id = ?", "keep = 1"]
    params: list = [dedup_run_id]
    if label_list:
        placeholders = ", ".join("?" for _ in label_list)
        clauses.append(f"predicted_label IN ({placeholders})")
        params.extend(label_list)
    rows = conn.execute(
        f"""
        SELECT result_id, image_rel_path, predicted_label
        FROM dedup_results
        WHERE {' AND '.join(clauses)}
        ORDER BY result_id
        """,
        params,
    ).fetchall()
    return [
        KeptBox(
            result_id=int(row["result_id"]),
            image_rel_path=str(row["image_rel_path"]),
            predicted_label=str(row["predicted_label"]),
        )
        for row in rows
    ]


def resolve_embedding_run_id(conn: sqlite3.Connection, requested: str) -> tuple[str, int]:
    raw = str(requested or "latest").strip()
    if raw and raw.lower() != "latest":
        row = conn.execute(
            "SELECT embedding_run_id, dim FROM embedding_runs WHERE embedding_run_id = ?",
            (raw,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Embedding run not found: {raw}")
        return str(row["embedding_run_id"]), int(row["dim"])
    row = conn.execute(
        "SELECT embedding_run_id, dim FROM embedding_runs ORDER BY created_at_utc DESC, embedding_run_id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("No embedding run found in embedding DB.")
    return str(row["embedding_run_id"]), int(row["dim"])


def load_embedding_matrix(
    conn: sqlite3.Connection,
    *,
    embedding_run_id: str,
    dim: int,
    boxes: list[KeptBox],
) -> tuple[np.ndarray, list[KeptBox]]:
    """Return (matrix [N, dim], boxes_with_embeddings).
    Boxes without embeddings are silently dropped."""
    if not boxes:
        return np.zeros((0, dim), dtype=np.float32), []

    by_id: dict[int, bytes] = {}
    ids = [int(b.result_id) for b in boxes]
    chunk = 900
    for offset in range(0, len(ids), chunk):
        slice_ids = ids[offset : offset + chunk]
        placeholders = ", ".join("?" for _ in slice_ids)
        rows = conn.execute(
            f"""
            SELECT result_id, embedding_blob
            FROM detection_embeddings
            WHERE embedding_run_id = ? AND result_id IN ({placeholders})
            """,
            [embedding_run_id, *slice_ids],
        ).fetchall()
        for row in rows:
            by_id[int(row["result_id"])] = bytes(row["embedding_blob"])

    matrix_rows: list[np.ndarray] = []
    kept: list[KeptBox] = []
    for box in boxes:
        blob = by_id.get(int(box.result_id))
        if blob is None:
            continue
        vec = np.frombuffer(blob, dtype="<f4")
        if vec.size != int(dim):
            continue
        matrix_rows.append(vec.astype(np.float32, copy=False))
        kept.append(box)
    if not matrix_rows:
        return np.zeros((0, dim), dtype=np.float32), []
    matrix = np.stack(matrix_rows, axis=0)
    return matrix, kept
