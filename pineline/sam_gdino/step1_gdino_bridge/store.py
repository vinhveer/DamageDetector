from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS run_info (
            run_id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            input_dir TEXT NOT NULL,
            checkpoint TEXT NOT NULL,
            device TEXT NOT NULL,
            box_threshold REAL NOT NULL,
            text_threshold REAL NOT NULL,
            score_floor REAL NOT NULL,
            top_k INTEGER NOT NULL,
            image_count INTEGER NOT NULL,
            kept_image_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS images (
            run_id TEXT NOT NULL,
            image_id TEXT NOT NULL,
            image_path TEXT NOT NULL,
            image_rel_path TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            kept INTEGER NOT NULL,
            best_score REAL NOT NULL,
            raw_box_count INTEGER NOT NULL,
            kept_box_count INTEGER NOT NULL,
            PRIMARY KEY (run_id, image_id)
        );

        CREATE TABLE IF NOT EXISTS bridge_detections (
            run_id TEXT NOT NULL,
            image_id TEXT NOT NULL,
            box_idx INTEGER NOT NULL,
            x1 REAL NOT NULL,
            y1 REAL NOT NULL,
            x2 REAL NOT NULL,
            y2 REAL NOT NULL,
            score REAL NOT NULL,
            label TEXT NOT NULL,
            PRIMARY KEY (run_id, image_id, box_idx)
        );

        CREATE INDEX IF NOT EXISTS idx_bridge_dets_image
            ON bridge_detections (run_id, image_id);
        """
    )
    conn.commit()


def insert_run_info(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO run_info
            (run_id, created_at_utc, input_dir, checkpoint, device,
             box_threshold, text_threshold, score_floor, top_k,
             image_count, kept_image_count)
        VALUES
            (:run_id, :created_at_utc, :input_dir, :checkpoint, :device,
             :box_threshold, :text_threshold, :score_floor, :top_k,
             :image_count, :kept_image_count)
        """,
        row,
    )
    conn.commit()


def insert_image(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO images
            (run_id, image_id, image_path, image_rel_path, width, height,
             kept, best_score, raw_box_count, kept_box_count)
        VALUES
            (:run_id, :image_id, :image_path, :image_rel_path, :width, :height,
             :kept, :best_score, :raw_box_count, :kept_box_count)
        """,
        row,
    )


def insert_bridge_detections(
    conn: sqlite3.Connection,
    rows: Iterable[dict],
) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO bridge_detections
            (run_id, image_id, box_idx, x1, y1, x2, y2, score, label)
        VALUES
            (:run_id, :image_id, :box_idx, :x1, :y1, :x2, :y2, :score, :label)
        """,
        rows,
    )
