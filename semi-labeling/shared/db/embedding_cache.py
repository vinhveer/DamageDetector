from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CropEmbeddingRow:
    result_id: int
    view_name: str
    crop_path: str
    crop_view_id: str


def latest_embedding_run(conn: sqlite3.Connection, *, model_name: str, view_name: str | None = None, run_id: str | None = None) -> sqlite3.Row | None:
    clauses = ["model_name = ?"]
    params: list[object] = [model_name]
    if view_name is not None:
        clauses.append("view_name = ?")
        params.append(view_name)
    if run_id is not None:
        clauses.append("run_id = ?")
        params.append(run_id)
    return conn.execute(
        f"""
        SELECT *
        FROM embedding_runs
        WHERE {' AND '.join(clauses)}
        ORDER BY created_at_utc DESC, embedding_run_id DESC
        LIMIT 1
        """,
        params,
    ).fetchone()


def load_embeddings(
    conn: sqlite3.Connection,
    *,
    model_name: str,
    view_name: str,
    run_id: str | None = None,
    embedding_run_id: str | None = None,
    result_ids: list[int] | None = None,
) -> tuple[np.ndarray, list[int], sqlite3.Row]:
    if embedding_run_id:
        run = conn.execute(
            "SELECT * FROM embedding_runs WHERE embedding_run_id = ?",
            (str(embedding_run_id),),
        ).fetchone()
    else:
        run = latest_embedding_run(conn, model_name=model_name, view_name=view_name, run_id=run_id)
    if run is None:
        suffix = f" run_id={run_id}" if run_id else ""
        raise RuntimeError(f"No embedding run found for model_name={model_name} view_name={view_name}{suffix}")
    embedding_run_id = str(run["embedding_run_id"])
    dim = int(run["dim"])
    if result_ids is None:
        rows = conn.execute(
            """
            SELECT result_id, embedding_blob
            FROM crop_embeddings
            WHERE embedding_run_id = ? AND view_name = ?
            ORDER BY result_id
            """,
            (embedding_run_id, view_name),
        ).fetchall()
    else:
        requested = [int(item) for item in result_ids]
        if not requested:
            return np.empty((0, dim), dtype=np.float32), [], run
        placeholders = ", ".join("?" for _ in requested)
        rows = conn.execute(
            f"""
            SELECT result_id, embedding_blob
            FROM crop_embeddings
            WHERE embedding_run_id = ? AND view_name = ? AND result_id IN ({placeholders})
            """,
            [embedding_run_id, view_name, *requested],
        ).fetchall()
        by_id = {int(row["result_id"]): row for row in rows}
        rows = [by_id[item] for item in requested if item in by_id]
    ids = [int(row["result_id"]) for row in rows]
    vectors = [_decode_embedding(bytes(row["embedding_blob"]), dim=dim, result_id=int(row["result_id"])) for row in rows]
    if not vectors:
        return np.empty((0, dim), dtype=np.float32), [], run
    return np.vstack(vectors).astype(np.float32, copy=False), ids, run


def read_crop_views(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    view_name: str,
    result_ids: list[int] | None = None,
    limit: int = 0,
) -> list[CropEmbeddingRow]:
    clauses = ["run_id = ?", "view_name = ?", "status = 'ok'", "crop_path IS NOT NULL", "crop_path != ''"]
    params: list[object] = [run_id, view_name]
    if result_ids:
        ids = [int(item) for item in result_ids]
        placeholders = ", ".join("?" for _ in ids)
        clauses.append(f"result_id IN ({placeholders})")
        params.extend(ids)
    sql = f"""
        SELECT result_id, view_name, crop_path
        FROM crop_views
        WHERE {' AND '.join(clauses)}
        ORDER BY result_id
    """
    if int(limit) > 0:
        sql = f"{sql} LIMIT {int(limit)}"
    rows = conn.execute(sql, params).fetchall()
    return [
        CropEmbeddingRow(
            result_id=int(row["result_id"]),
            view_name=str(row["view_name"]),
            crop_path=str(row["crop_path"]),
            crop_view_id=f"{run_id}:{int(row['result_id'])}:{str(row['view_name'])}",
        )
        for row in rows
    ]


def existing_embedding_keys(conn: sqlite3.Connection, *, embedding_run_id: str) -> set[tuple[int, str]]:
    rows = conn.execute(
        "SELECT result_id, view_name FROM crop_embeddings WHERE embedding_run_id = ?",
        (embedding_run_id,),
    ).fetchall()
    return {(int(row["result_id"]), str(row["view_name"])) for row in rows}


def existing_skip_keys(conn: sqlite3.Connection, *, embedding_run_id: str) -> set[tuple[int, str]]:
    rows = conn.execute(
        "SELECT result_id, view_name FROM skipped_crop_embeddings WHERE embedding_run_id = ?",
        (embedding_run_id,),
    ).fetchall()
    return {(int(row["result_id"]), str(row["view_name"])) for row in rows}


def update_embedding_counts(conn: sqlite3.Connection, *, embedding_run_id: str) -> tuple[int, int]:
    embedded = int(conn.execute("SELECT COUNT(*) FROM crop_embeddings WHERE embedding_run_id = ?", (embedding_run_id,)).fetchone()[0])
    skipped = int(conn.execute("SELECT COUNT(*) FROM skipped_crop_embeddings WHERE embedding_run_id = ?", (embedding_run_id,)).fetchone()[0])
    conn.execute(
        "UPDATE embedding_runs SET embedded_count = ?, skipped_count = ? WHERE embedding_run_id = ?",
        (embedded, skipped, embedding_run_id),
    )
    conn.commit()
    return embedded, skipped


def _decode_embedding(blob: bytes, *, dim: int, result_id: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype="<f4")
    if arr.size != int(dim):
        raise ValueError(f"Invalid embedding size for result_id={result_id}: got {arr.size}, expected {dim}")
    return arr.astype(np.float32, copy=True)
