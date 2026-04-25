from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import Detection, ImageInfo


class DamageScanStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.ensure_schema()

    def close(self) -> None:
        with self._lock:
            self.conn.commit()
            self.conn.close()

    def ensure_schema(self) -> None:
        with self._lock:
            self.conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                PRAGMA foreign_keys=ON;

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    input_dir TEXT NOT NULL,
                    db_path TEXT NOT NULL,
                    detector_name TEXT NOT NULL,
                    checkpoint TEXT NOT NULL,
                    device TEXT NOT NULL,
                    config_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS images (
                    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    rel_path TEXT NOT NULL,
                    path TEXT NOT NULL,
                    name TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    error_type TEXT,
                    error_message TEXT,
                    UNIQUE(run_id, rel_path),
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS detections (
                    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    image_id INTEGER NOT NULL,
                    parent_detection_id INTEGER,
                    stage TEXT NOT NULL,
                    source TEXT NOT NULL,
                    prompt_key TEXT NOT NULL,
                    prompt_text TEXT NOT NULL,
                    label TEXT NOT NULL,
                    score REAL NOT NULL,
                    x1 REAL NOT NULL,
                    y1 REAL NOT NULL,
                    x2 REAL NOT NULL,
                    y2 REAL NOT NULL,
                    box_w REAL NOT NULL,
                    box_h REAL NOT NULL,
                    area_px2 REAL NOT NULL,
                    model_name TEXT NOT NULL,
                    raw_json TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id),
                    FOREIGN KEY(image_id) REFERENCES images(image_id),
                    FOREIGN KEY(parent_detection_id) REFERENCES detections(detection_id)
                );

                CREATE INDEX IF NOT EXISTS idx_damage_scan_detections_image
                ON detections(run_id, image_id, stage, prompt_key);

                CREATE INDEX IF NOT EXISTS idx_damage_scan_detections_parent
                ON detections(parent_detection_id);
                """
            )
            self.conn.commit()

    def create_run(
        self,
        *,
        run_id: str,
        input_dir: Path,
        detector_name: str,
        checkpoint: str,
        device: str,
        config: dict[str, Any],
    ) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, created_at_utc, input_dir, db_path, detector_name, checkpoint, device, config_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                    str(input_dir),
                    str(self.db_path),
                    detector_name,
                    checkpoint,
                    device,
                    json.dumps(config, ensure_ascii=False, sort_keys=True),
                ),
            )
            self.conn.commit()

    def upsert_image(self, *, run_id: str, image: ImageInfo, status: str = "pending") -> int:
        with self._lock:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO images (run_id, rel_path, path, name, width, height, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, image.rel_path, str(image.path), image.path.name, int(image.width), int(image.height), status),
            )
            self.conn.execute(
                """
                UPDATE images
                SET path = ?, name = ?, width = ?, height = ?, status = ?
                WHERE run_id = ? AND rel_path = ?
                """,
                (str(image.path), image.path.name, int(image.width), int(image.height), status, run_id, image.rel_path),
            )
            row = self.conn.execute(
                "SELECT image_id FROM images WHERE run_id = ? AND rel_path = ?",
                (run_id, image.rel_path),
            ).fetchone()
            self.conn.commit()
            return int(row["image_id"])

    def mark_image_error(self, *, image_id: int, error: Exception) -> None:
        with self._lock:
            self.conn.execute(
                """
                UPDATE images SET status = 'error', error_type = ?, error_message = ? WHERE image_id = ?
                """,
                (error.__class__.__name__, str(error), int(image_id)),
            )
            self.conn.commit()

    def mark_image_done(self, *, image_id: int) -> None:
        with self._lock:
            self.conn.execute("UPDATE images SET status = 'ok', error_type = NULL, error_message = NULL WHERE image_id = ?", (int(image_id),))
            self.conn.commit()

    def insert_detection(self, *, run_id: str, image_id: int, detection: Detection) -> int:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO detections (
                    run_id, image_id, parent_detection_id, stage, source, prompt_key, prompt_text,
                    label, score, x1, y1, x2, y2, box_w, box_h, area_px2, model_name, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    int(image_id),
                    detection.parent_detection_id,
                    detection.stage,
                    detection.source,
                    detection.prompt_key,
                    detection.prompt_text,
                    detection.label,
                    float(detection.score),
                    float(detection.box.x1),
                    float(detection.box.y1),
                    float(detection.box.x2),
                    float(detection.box.y2),
                    float(detection.box.width),
                    float(detection.box.height),
                    float(detection.box.area),
                    detection.model_name,
                    json.dumps(detection.raw, ensure_ascii=False, sort_keys=True),
                ),
            )
            self.conn.commit()
            return int(self.conn.execute("SELECT last_insert_rowid()").fetchone()[0])

    def insert_detections_bulk(self, *, run_id: str, image_id: int, detections: list[Detection]) -> int:
        if not detections:
            return 0
        rows = [
            (
                run_id,
                int(image_id),
                detection.parent_detection_id,
                detection.stage,
                detection.source,
                detection.prompt_key,
                detection.prompt_text,
                detection.label,
                float(detection.score),
                float(detection.box.x1),
                float(detection.box.y1),
                float(detection.box.x2),
                float(detection.box.y2),
                float(detection.box.width),
                float(detection.box.height),
                float(detection.box.area),
                detection.model_name,
                json.dumps(detection.raw, ensure_ascii=False, sort_keys=True),
            )
            for detection in detections
        ]
        with self._lock:
            self.conn.executemany(
                """
                INSERT INTO detections (
                    run_id, image_id, parent_detection_id, stage, source, prompt_key, prompt_text,
                    label, score, x1, y1, x2, y2, box_w, box_h, area_px2, model_name, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self.conn.commit()
        return len(rows)
