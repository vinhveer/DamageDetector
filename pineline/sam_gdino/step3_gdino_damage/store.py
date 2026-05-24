from __future__ import annotations

import sqlite3
from typing import Iterable


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS step3_run_info (
            run_id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            step2_db TEXT NOT NULL,
            step2_run_id TEXT NOT NULL,
            checkpoint TEXT NOT NULL,
            device TEXT NOT NULL,
            box_threshold REAL NOT NULL,
            text_threshold REAL NOT NULL,
            max_dets INTEGER NOT NULL,
            crop_count INTEGER NOT NULL,
            detection_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS step3_prompt_groups (
            run_id TEXT NOT NULL,
            group_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            slug TEXT NOT NULL,
            queries TEXT NOT NULL,
            PRIMARY KEY (run_id, group_id)
        );

        CREATE TABLE IF NOT EXISTS damage_detections (
            run_id TEXT NOT NULL,
            parent_image_id TEXT NOT NULL,
            det_idx INTEGER NOT NULL,
            group_id INTEGER NOT NULL,
            group_name TEXT NOT NULL,
            label TEXT NOT NULL,
            x1 REAL NOT NULL,
            y1 REAL NOT NULL,
            x2 REAL NOT NULL,
            y2 REAL NOT NULL,
            score REAL NOT NULL,
            PRIMARY KEY (run_id, parent_image_id, det_idx)
        );

        CREATE INDEX IF NOT EXISTS idx_damage_parent
            ON damage_detections (run_id, parent_image_id);
        CREATE INDEX IF NOT EXISTS idx_damage_group
            ON damage_detections (run_id, group_id);
        """
    )
    conn.commit()


def insert_run_info(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO step3_run_info
            (run_id, created_at_utc, step2_db, step2_run_id,
             checkpoint, device, box_threshold, text_threshold, max_dets,
             crop_count, detection_count)
        VALUES
            (:run_id, :created_at_utc, :step2_db, :step2_run_id,
             :checkpoint, :device, :box_threshold, :text_threshold, :max_dets,
             :crop_count, :detection_count)
        """,
        row,
    )
    conn.commit()


def insert_prompt_groups(conn: sqlite3.Connection, run_id: str, groups: list) -> None:
    rows = [
        {
            "run_id": run_id,
            "group_id": g.group_id,
            "name": g.name,
            "slug": g.slug,
            "queries": ",".join(g.queries),
        }
        for g in groups
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO step3_prompt_groups
            (run_id, group_id, name, slug, queries)
        VALUES (:run_id, :group_id, :name, :slug, :queries)
        """,
        rows,
    )
    conn.commit()


def insert_damage_detections(conn: sqlite3.Connection, rows: Iterable[dict]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO damage_detections
            (run_id, parent_image_id, det_idx,
             group_id, group_name, label,
             x1, y1, x2, y2, score)
        VALUES
            (:run_id, :parent_image_id, :det_idx,
             :group_id, :group_name, :label,
             :x1, :y1, :x2, :y2, :score)
        """,
        rows,
    )


def load_step2_crops(
    step2_conn: sqlite3.Connection,
    *,
    source_run_id: str | None = None,
) -> list[dict]:
    if source_run_id is None or str(source_run_id).strip().lower() == "latest":
        cur = step2_conn.execute(
            "SELECT run_id FROM step2_run_info ORDER BY created_at_utc DESC LIMIT 1"
        )
        row = cur.fetchone()
        if row is None:
            return []
        source_run_id = row[0]
    cur = step2_conn.execute(
        """
        SELECT parent_image_id, parent_image_path, crop_path, mask_path,
               crop_x1, crop_y1, crop_x2, crop_y2
        FROM bridge_crops
        WHERE run_id = ?
        ORDER BY parent_image_id
        """,
        (source_run_id,),
    )
    return [
        {
            "parent_image_id": r[0],
            "parent_image_path": r[1],
            "crop_path": r[2],
            "mask_path": r[3],
            "crop_x1": int(r[4]), "crop_y1": int(r[5]),
            "crop_x2": int(r[6]), "crop_y2": int(r[7]),
            "step2_run_id": source_run_id,
        }
        for r in cur.fetchall()
    ]
