from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from models import KeptBox


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.expanduser().resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=60.0)
    conn.row_factory = sqlite3.Row
    return conn


def resolve_filter_run_id(filtered_db_path: Path, requested: str) -> str:
    raw = str(requested or "latest").strip()
    if raw and raw.lower() != "latest":
        return raw
    conn = connect_readonly(filtered_db_path)
    try:
        row = conn.execute("SELECT filter_run_id FROM filter_runs ORDER BY created_at_utc DESC LIMIT 1").fetchone()
    finally:
        conn.close()
    if row is None:
        raise RuntimeError("No filter run found in filtered SQLite DB.")
    return str(row["filter_run_id"])


def load_kept_boxes(
    *,
    source_db_path: Path,
    filtered_db_path: Path,
    filter_run_id: str,
    labels: tuple[str, ...],
    limit: int,
) -> list[KeptBox]:
    conn = connect_readonly(filtered_db_path)
    conn.execute("ATTACH DATABASE ? AS src", (str(source_db_path.expanduser().resolve()),))
    try:
        clauses = ["fr.filter_run_id = ?", "fr.keep = 1"]
        params: list[Any] = [filter_run_id]
        if labels:
            placeholders = ", ".join("?" for _ in labels)
            clauses.append(f"res.predicted_label IN ({placeholders})")
            params.extend(labels)
        sql = f"""
            SELECT res.result_id, res.source_detection_id, res.image_id, res.image_rel_path,
                   res.image_path, src_run.input_dir AS source_input_dir,
                   res.predicted_label, res.predicted_probability_pct, res.detector_score,
                   res.x1, res.y1, res.x2, res.y2,
                   img.width AS image_width, img.height AS image_height,
                   fr.filter_run_id
            FROM filter_results fr
            JOIN src.openclip_semantic_results res ON res.result_id = fr.result_id
            JOIN src.images img ON img.image_id = res.image_id
            JOIN src.runs src_run ON src_run.run_id = res.source_run_id
            WHERE {' AND '.join(clauses)}
            ORDER BY res.predicted_label, res.image_rel_path, res.result_id
        """
        if int(limit) > 0:
            sql = f"{sql} LIMIT {int(limit)}"
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [KeptBox(**dict(row)) for row in rows]
