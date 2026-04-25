from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from prompts import PromptGroup


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS run_info (
            run_id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            input_dir TEXT NOT NULL,
            output_dir TEXT NOT NULL,
            checkpoint TEXT NOT NULL,
            device TEXT NOT NULL,
            box_threshold REAL NOT NULL,
            text_threshold REAL NOT NULL,
            max_dets INTEGER NOT NULL,
            tiled_threshold INTEGER NOT NULL,
            recursive_find INTEGER NOT NULL,
            recursive_max_depth INTEGER NOT NULL,
            recursive_min_box_px INTEGER NOT NULL,
            image_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS request_aliases (
            run_id TEXT NOT NULL,
            request_name TEXT NOT NULL,
            engine_name TEXT NOT NULL,
            notes TEXT NOT NULL,
            PRIMARY KEY (run_id, request_name)
        );

        CREATE TABLE IF NOT EXISTS prompt_groups (
            run_id TEXT NOT NULL,
            prompt_group_id INTEGER NOT NULL,
            prompt_group_name TEXT NOT NULL,
            prompt_group_slug TEXT NOT NULL,
            prompt_text TEXT NOT NULL,
            PRIMARY KEY (run_id, prompt_group_id)
        );

        CREATE TABLE IF NOT EXISTS images (
            run_id TEXT NOT NULL,
            image_rel_path TEXT NOT NULL,
            image_path TEXT NOT NULL,
            image_name TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            max_dim INTEGER NOT NULL,
            PRIMARY KEY (run_id, image_rel_path)
        );

        CREATE TABLE IF NOT EXISTS image_runs (
            run_id TEXT NOT NULL,
            prompt_group_id INTEGER NOT NULL,
            image_rel_path TEXT NOT NULL,
            mode TEXT NOT NULL,
            artifact_dir TEXT NOT NULL,
            status TEXT NOT NULL,
            detection_count INTEGER NOT NULL,
            error_type TEXT,
            error_message TEXT,
            PRIMARY KEY (run_id, prompt_group_id, image_rel_path)
        );

        CREATE TABLE IF NOT EXISTS detections (
            run_id TEXT NOT NULL,
            request_name TEXT NOT NULL,
            engine_name TEXT NOT NULL,
            prompt_group_id INTEGER NOT NULL,
            image_rel_path TEXT NOT NULL,
            image_path TEXT NOT NULL,
            image_name TEXT NOT NULL,
            label TEXT NOT NULL,
            score REAL NOT NULL,
            x1 INTEGER NOT NULL,
            y1 INTEGER NOT NULL,
            x2 INTEGER NOT NULL,
            y2 INTEGER NOT NULL,
            box_w INTEGER NOT NULL,
            box_h INTEGER NOT NULL,
            area_px2 INTEGER NOT NULL,
            artifact_dir TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_detections_lookup
        ON detections (run_id, request_name, prompt_group_id, image_rel_path);

        CREATE INDEX IF NOT EXISTS idx_image_runs_status
        ON image_runs (run_id, prompt_group_id, status);
        """
    )
    conn.commit()


def insert_run_metadata(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    input_dir: Path,
    output_dir: Path,
    checkpoint: str,
    device: str,
    box_threshold: float,
    text_threshold: float,
    max_dets: int,
    tiled_threshold: int,
    recursive_find: bool,
    recursive_max_depth: int,
    recursive_min_box_px: int,
    image_count: int,
    request_names: list[str],
    prompt_groups: list[PromptGroup],
) -> None:
    created_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    conn.execute(
        """
        INSERT OR REPLACE INTO run_info (
            run_id, created_at_utc, input_dir, output_dir, checkpoint, device,
            box_threshold, text_threshold, max_dets, tiled_threshold,
            recursive_find, recursive_max_depth, recursive_min_box_px, image_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            created_at_utc,
            str(input_dir),
            str(output_dir),
            checkpoint,
            device,
            float(box_threshold),
            float(text_threshold),
            int(max_dets),
            int(tiled_threshold),
            1 if recursive_find else 0,
            int(recursive_max_depth),
            int(recursive_min_box_px),
            int(image_count),
        ),
    )
    for request_name in request_names:
        conn.execute(
            """
            INSERT OR REPLACE INTO request_aliases (run_id, request_name, engine_name, notes)
            VALUES (?, ?, ?, ?)
            """,
            (
                run_id,
                request_name,
                "groundingdino",
                "This repository routes prompt-based DINO detection through the GroundingDINO engine.",
            ),
        )
    for group in prompt_groups:
        conn.execute(
            """
            INSERT OR REPLACE INTO prompt_groups (
                run_id, prompt_group_id, prompt_group_name, prompt_group_slug, prompt_text
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, int(group.group_id), group.name, group.slug, group.prompt_text),
        )
    conn.commit()


def insert_image_metadata(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    image_rel_path: str,
    image_path: Path,
    width: int,
    height: int,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO images (
            run_id, image_rel_path, image_path, image_name, width, height, max_dim
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            image_rel_path,
            str(image_path),
            image_path.name,
            int(width),
            int(height),
            int(max(width, height)),
        ),
    )


def record_image_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    prompt_group_id: int,
    image_rel_path: str,
    mode: str,
    artifact_dir: Path,
    status: str,
    detection_count: int,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO image_runs (
            run_id, prompt_group_id, image_rel_path, mode, artifact_dir, status,
            detection_count, error_type, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            int(prompt_group_id),
            image_rel_path,
            mode,
            str(artifact_dir),
            status,
            int(detection_count),
            error_type,
            error_message,
        ),
    )


def delete_existing_detections(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    prompt_group_id: int,
    image_rel_path: str,
) -> None:
    conn.execute(
        """
        DELETE FROM detections
        WHERE run_id = ? AND prompt_group_id = ? AND image_rel_path = ?
        """,
        (run_id, int(prompt_group_id), image_rel_path),
    )


def insert_detections(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    request_names: list[str],
    prompt_group_id: int,
    image_rel_path: str,
    image_path: Path,
    rows: Iterable,
    artifact_dir: Path,
) -> None:
    payload = []
    for row in rows:
        for request_name in request_names:
            payload.append(
                (
                    run_id,
                    request_name,
                    "groundingdino",
                    int(prompt_group_id),
                    image_rel_path,
                    str(image_path),
                    image_path.name,
                    row.label,
                    float(row.score),
                    int(row.x1),
                    int(row.y1),
                    int(row.x2),
                    int(row.y2),
                    int(row.w),
                    int(row.h),
                    int(row.area_px2),
                    str(artifact_dir),
                )
            )
    if payload:
        conn.executemany(
            """
            INSERT INTO detections (
                run_id, request_name, engine_name, prompt_group_id, image_rel_path, image_path,
                image_name, label, score, x1, y1, x2, y2, box_w, box_h, area_px2, artifact_dir
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
