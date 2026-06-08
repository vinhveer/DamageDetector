from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            source_db TEXT NOT NULL,
            input_dir TEXT NOT NULL,
            output_root TEXT NOT NULL,
            device TEXT NOT NULL,
            stable_dino_dir TEXT NOT NULL,
            yolo_model TEXT NOT NULL,
            sam_model_dir TEXT NOT NULL,
            unet_model TEXT NOT NULL,
            conf_threshold REAL NOT NULL,
            nms_iou REAL NOT NULL,
            image_ids_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS images (
            run_id TEXT NOT NULL,
            image_id INTEGER NOT NULL,
            rel_path TEXT NOT NULL,
            image_path TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            PRIMARY KEY (run_id, image_id)
        );

        CREATE TABLE IF NOT EXISTS detections (
            run_id TEXT NOT NULL,
            pipeline_name TEXT NOT NULL,
            image_id INTEGER NOT NULL,
            det_idx INTEGER NOT NULL,
            detector_name TEXT NOT NULL,
            label TEXT NOT NULL,
            score REAL NOT NULL,
            x1 REAL NOT NULL,
            y1 REAL NOT NULL,
            x2 REAL NOT NULL,
            y2 REAL NOT NULL,
            PRIMARY KEY (run_id, pipeline_name, image_id, det_idx)
        );

        CREATE TABLE IF NOT EXISTS segmentations (
            run_id TEXT NOT NULL,
            pipeline_name TEXT NOT NULL,
            image_id INTEGER NOT NULL,
            det_idx INTEGER NOT NULL,
            segmenter_name TEXT NOT NULL,
            mask_path TEXT,
            mask_area_px INTEGER NOT NULL,
            score REAL,
            extra_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY (run_id, pipeline_name, image_id, det_idx, segmenter_name)
        );

        CREATE TABLE IF NOT EXISTS artifacts (
            run_id TEXT NOT NULL,
            pipeline_name TEXT NOT NULL,
            image_id INTEGER NOT NULL,
            artifact_type TEXT NOT NULL,
            path TEXT NOT NULL,
            PRIMARY KEY (run_id, pipeline_name, image_id, artifact_type)
        );
        """
    )
    conn.commit()


def insert_run(conn: sqlite3.Connection, row: dict) -> None:
    payload = dict(row)
    payload["image_ids_json"] = json.dumps(list(payload["image_ids_json"]))
    conn.execute(
        """
        INSERT OR REPLACE INTO runs
            (run_id, created_at_utc, source_db, input_dir, output_root, device,
             stable_dino_dir, yolo_model, sam_model_dir, unet_model,
             conf_threshold, nms_iou, image_ids_json)
        VALUES
            (:run_id, :created_at_utc, :source_db, :input_dir, :output_root, :device,
             :stable_dino_dir, :yolo_model, :sam_model_dir, :unet_model,
             :conf_threshold, :nms_iou, :image_ids_json)
        """,
        payload,
    )
    conn.commit()


def insert_images(conn: sqlite3.Connection, rows: Iterable[dict]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO images
            (run_id, image_id, rel_path, image_path, width, height)
        VALUES
            (:run_id, :image_id, :rel_path, :image_path, :width, :height)
        """,
        list(rows),
    )
    conn.commit()


def insert_detections(conn: sqlite3.Connection, rows: Iterable[dict]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO detections
            (run_id, pipeline_name, image_id, det_idx, detector_name, label, score, x1, y1, x2, y2)
        VALUES
            (:run_id, :pipeline_name, :image_id, :det_idx, :detector_name, :label,
             :score, :x1, :y1, :x2, :y2)
        """,
        list(rows),
    )
    conn.commit()


def insert_segmentations(conn: sqlite3.Connection, rows: Iterable[dict]) -> None:
    payload = []
    for row in rows:
        item = dict(row)
        extra = item.get("extra_json", {})
        item["extra_json"] = extra if isinstance(extra, str) else json.dumps(extra, ensure_ascii=False)
        payload.append(item)
    conn.executemany(
        """
        INSERT OR REPLACE INTO segmentations
            (run_id, pipeline_name, image_id, det_idx, segmenter_name, mask_path,
             mask_area_px, score, extra_json)
        VALUES
            (:run_id, :pipeline_name, :image_id, :det_idx, :segmenter_name, :mask_path,
             :mask_area_px, :score, :extra_json)
        """,
        payload,
    )
    conn.commit()


def insert_artifacts(conn: sqlite3.Connection, rows: Iterable[dict]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO artifacts
            (run_id, pipeline_name, image_id, artifact_type, path)
        VALUES
            (:run_id, :pipeline_name, :image_id, :artifact_type, :path)
        """,
        list(rows),
    )
    conn.commit()
