from __future__ import annotations

import sqlite3
from typing import Iterable


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS step2_run_info (
            run_id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            source_db TEXT NOT NULL,
            source_run_id TEXT NOT NULL,
            sam_checkpoint TEXT NOT NULL,
            sam_model_type TEXT NOT NULL,
            device TEXT NOT NULL,
            points_per_box INTEGER NOT NULL,
            pad_px INTEGER NOT NULL,
            image_count INTEGER NOT NULL,
            crop_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS bridge_crops (
            run_id TEXT NOT NULL,
            parent_image_id TEXT NOT NULL,
            parent_image_path TEXT NOT NULL,
            crop_path TEXT NOT NULL,
            mask_path TEXT NOT NULL,
            crop_x1 INTEGER NOT NULL,
            crop_y1 INTEGER NOT NULL,
            crop_x2 INTEGER NOT NULL,
            crop_y2 INTEGER NOT NULL,
            mask_area_px INTEGER NOT NULL,
            source_box_count INTEGER NOT NULL,
            PRIMARY KEY (run_id, parent_image_id)
        );

        CREATE INDEX IF NOT EXISTS idx_crops_parent
            ON bridge_crops (run_id, parent_image_id);
        """
    )
    conn.commit()


def insert_run_info(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO step2_run_info
            (run_id, created_at_utc, source_db, source_run_id,
             sam_checkpoint, sam_model_type, device,
             points_per_box, pad_px, image_count, crop_count)
        VALUES
            (:run_id, :created_at_utc, :source_db, :source_run_id,
             :sam_checkpoint, :sam_model_type, :device,
             :points_per_box, :pad_px, :image_count, :crop_count)
        """,
        row,
    )
    conn.commit()


def insert_crop(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO bridge_crops
            (run_id, parent_image_id, parent_image_path,
             crop_path, mask_path,
             crop_x1, crop_y1, crop_x2, crop_y2,
             mask_area_px, source_box_count)
        VALUES
            (:run_id, :parent_image_id, :parent_image_path,
             :crop_path, :mask_path,
             :crop_x1, :crop_y1, :crop_x2, :crop_y2,
             :mask_area_px, :source_box_count)
        """,
        row,
    )


def load_kept_images(
    step1_db: sqlite3.Connection,
    *,
    source_run_id: str | None = None,
) -> list[dict]:
    """Return one row per kept image with its boxes attached."""
    if source_run_id is None or str(source_run_id).strip().lower() == "latest":
        cur = step1_db.execute(
            "SELECT run_id FROM run_info ORDER BY created_at_utc DESC LIMIT 1"
        )
        row = cur.fetchone()
        if row is None:
            return []
        source_run_id = row[0]
    cur = step1_db.execute(
        """
        SELECT image_id, image_path, image_rel_path, width, height
        FROM images
        WHERE run_id = ? AND kept = 1
        ORDER BY image_rel_path
        """,
        (source_run_id,),
    )
    images = [
        {
            "image_id": r[0],
            "image_path": r[1],
            "image_rel_path": r[2],
            "width": int(r[3]),
            "height": int(r[4]),
            "source_run_id": source_run_id,
            "boxes": [],
        }
        for r in cur.fetchall()
    ]
    if not images:
        return []
    placeholders = ",".join("?" for _ in images)
    cur = step1_db.execute(
        f"""
        SELECT image_id, box_idx, x1, y1, x2, y2, score
        FROM bridge_detections
        WHERE run_id = ? AND image_id IN ({placeholders})
        ORDER BY image_id, box_idx
        """,
        (source_run_id, *[img["image_id"] for img in images]),
    )
    by_id = {img["image_id"]: img for img in images}
    for image_id, box_idx, x1, y1, x2, y2, score in cur.fetchall():
        by_id[image_id]["boxes"].append(
            {
                "box_idx": int(box_idx),
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "score": float(score),
            }
        )
    return [img for img in images if img["boxes"]]
