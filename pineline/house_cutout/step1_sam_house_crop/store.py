from __future__ import annotations

import sqlite3
from typing import Iterable


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS run_info (
            run_id           TEXT PRIMARY KEY,
            created_at_utc   TEXT,
            input_dir        TEXT,
            gdino_checkpoint TEXT,
            sam_checkpoint   TEXT,
            device           TEXT,
            work_max_side    INTEGER,
            box_threshold    REAL,
            text_threshold   REAL,
            score_floor      REAL,
            points_per_box   INTEGER,
            pad_px           INTEGER,
            image_count      INTEGER,
            cutout_count     INTEGER
        );

        CREATE TABLE IF NOT EXISTS images (
            run_id          TEXT,
            image_id        TEXT,
            image_path      TEXT,
            image_rel_path  TEXT,
            orig_width      INTEGER,
            orig_height     INTEGER,
            work_width      INTEGER,
            work_height     INTEGER,
            scale           REAL,
            house_box_count INTEGER,
            neg_box_count   INTEGER,
            has_cutout      INTEGER,
            PRIMARY KEY (run_id, image_id)
        );

        CREATE TABLE IF NOT EXISTS detections (
            run_id    TEXT,
            image_id  TEXT,
            det_idx   INTEGER,
            role      TEXT,        -- 'house' | 'negative'
            label     TEXT,
            x1 REAL, y1 REAL, x2 REAL, y2 REAL,
            score     REAL,
            PRIMARY KEY (run_id, image_id, det_idx)
        );

        CREATE TABLE IF NOT EXISTS crops (
            run_id           TEXT,
            image_id         TEXT,
            cutout_path      TEXT,
            mask_path        TEXT,
            crop_x1 INTEGER, crop_y1 INTEGER, crop_x2 INTEGER, crop_y2 INTEGER,
            mask_area_px     INTEGER,
            source_box_count INTEGER,
            PRIMARY KEY (run_id, image_id)
        );

        CREATE INDEX IF NOT EXISTS idx_det_image
            ON detections (run_id, image_id);
        """
    )
    conn.commit()


def insert_run_info(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO run_info
            (run_id, created_at_utc, input_dir, gdino_checkpoint, sam_checkpoint,
             device, work_max_side, box_threshold, text_threshold, score_floor,
             points_per_box, pad_px, image_count, cutout_count)
        VALUES
            (:run_id, :created_at_utc, :input_dir, :gdino_checkpoint, :sam_checkpoint,
             :device, :work_max_side, :box_threshold, :text_threshold, :score_floor,
             :points_per_box, :pad_px, :image_count, :cutout_count)
        """,
        row,
    )
    conn.commit()


def insert_image(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO images
            (run_id, image_id, image_path, image_rel_path,
             orig_width, orig_height, work_width, work_height, scale,
             house_box_count, neg_box_count, has_cutout)
        VALUES
            (:run_id, :image_id, :image_path, :image_rel_path,
             :orig_width, :orig_height, :work_width, :work_height, :scale,
             :house_box_count, :neg_box_count, :has_cutout)
        """,
        row,
    )


def insert_detections(conn: sqlite3.Connection, rows: Iterable[dict]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO detections
            (run_id, image_id, det_idx, role, label, x1, y1, x2, y2, score)
        VALUES
            (:run_id, :image_id, :det_idx, :role, :label, :x1, :y1, :x2, :y2, :score)
        """,
        rows,
    )


def insert_crop(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO crops
            (run_id, image_id, cutout_path, mask_path,
             crop_x1, crop_y1, crop_x2, crop_y2, mask_area_px, source_box_count)
        VALUES
            (:run_id, :image_id, :cutout_path, :mask_path,
             :crop_x1, :crop_y1, :crop_x2, :crop_y2, :mask_area_px, :source_box_count)
        """,
        row,
    )
