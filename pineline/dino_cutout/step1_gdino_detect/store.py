from __future__ import annotations

import sqlite3
from typing import Iterable


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS run_info (
            run_id          TEXT PRIMARY KEY,
            created_at_utc  TEXT,
            input_dir       TEXT,
            checkpoint      TEXT,
            device          TEXT,
            box_threshold   REAL,
            text_threshold  REAL,
            tile_scales     TEXT,
            max_dets        INTEGER,
            image_count     INTEGER,
            detection_count INTEGER
        );

        CREATE TABLE IF NOT EXISTS images (
            run_id          TEXT,
            image_id        TEXT,
            image_path      TEXT,
            image_rel_path  TEXT,
            width           INTEGER,
            height          INTEGER,
            orig_width      INTEGER,
            orig_height     INTEGER,
            offset_x        INTEGER,
            offset_y        INTEGER,
            det_count       INTEGER,
            PRIMARY KEY (run_id, image_id)
        );

        CREATE TABLE IF NOT EXISTS detections (
            run_id          TEXT,
            image_id        TEXT,
            det_idx         INTEGER,
            group_id        INTEGER,
            group_name      TEXT,
            label           TEXT,
            query_label     TEXT,
            x1 REAL, y1 REAL, x2 REAL, y2 REAL,
            score           REAL,
            PRIMARY KEY (run_id, image_id, det_idx)
        );

        CREATE TABLE IF NOT EXISTS prompt_groups (
            run_id       TEXT,
            group_id     INTEGER,
            name         TEXT,
            slug         TEXT,
            queries_json TEXT,
            PRIMARY KEY (run_id, group_id)
        );

        CREATE INDEX IF NOT EXISTS idx_detections_image
            ON detections (run_id, image_id);
        CREATE INDEX IF NOT EXISTS idx_detections_group
            ON detections (run_id, group_id);
        """
    )
    conn.commit()


def insert_run_info(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO run_info
            (run_id, created_at_utc, input_dir, checkpoint, device,
             box_threshold, text_threshold, tile_scales, max_dets,
             image_count, detection_count)
        VALUES
            (:run_id, :created_at_utc, :input_dir, :checkpoint, :device,
             :box_threshold, :text_threshold, :tile_scales, :max_dets,
             :image_count, :detection_count)
        """,
        row,
    )
    conn.commit()


def insert_prompt_groups(conn: sqlite3.Connection, run_id: str, groups: list) -> None:
    import json

    rows = [
        {
            "run_id": run_id,
            "group_id": g.group_id,
            "name": g.name,
            "slug": g.slug,
            "queries_json": json.dumps(list(g.queries)),
        }
        for g in groups
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO prompt_groups
            (run_id, group_id, name, slug, queries_json)
        VALUES (:run_id, :group_id, :name, :slug, :queries_json)
        """,
        rows,
    )
    conn.commit()


def insert_images(conn: sqlite3.Connection, rows: Iterable[dict]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO images
            (run_id, image_id, image_path, image_rel_path,
             width, height, orig_width, orig_height,
             offset_x, offset_y, det_count)
        VALUES
            (:run_id, :image_id, :image_path, :image_rel_path,
             :width, :height, :orig_width, :orig_height,
             :offset_x, :offset_y, :det_count)
        """,
        rows,
    )


def insert_detections(conn: sqlite3.Connection, rows: Iterable[dict]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO detections
            (run_id, image_id, det_idx,
             group_id, group_name, label, query_label,
             x1, y1, x2, y2, score)
        VALUES
            (:run_id, :image_id, :det_idx,
             :group_id, :group_name, :label, :query_label,
             :x1, :y1, :x2, :y2, :score)
        """,
        rows,
    )
