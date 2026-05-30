from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import Detection, ImageInfo


# --- C1: geometry columns (additive, idempotent migration) -------------------

# Ordered (column, DDL); read defaults derive from this list (counts -> 0, else 0.0).
_GEOMETRY_DDL: tuple[tuple[str, str], ...] = (
    ("box_width", "REAL"),
    ("box_height", "REAL"),
    ("box_area", "REAL"),
    ("area_ratio_to_image", "REAL"),
    ("aspect_ratio", "REAL"),
    ("elongation", "REAL"),
    ("center_x", "REAL"),
    ("center_y", "REAL"),
    ("contains_count", "INTEGER DEFAULT 0"),
    ("contained_by_count", "INTEGER DEFAULT 0"),
    ("max_iou_same_label", "REAL DEFAULT 0.0"),
    ("max_containment", "REAL DEFAULT 0.0"),
)
_GEOMETRY_COLUMNS: tuple[str, ...] = tuple(name for name, _ in _GEOMETRY_DDL)
_GEOMETRY_DEFAULTS: dict[str, Any] = {
    name: (0 if name.endswith("_count") else 0.0) for name in _GEOMETRY_COLUMNS
}


def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def ensure_geometry_columns(conn: sqlite3.Connection) -> None:
    """Idempotent: add the 12 geometry columns to ``detections`` if absent."""
    existing = _existing_columns(conn, "detections")
    for name, decl in _GEOMETRY_DDL:
        if name not in existing:
            conn.execute(f"ALTER TABLE detections ADD COLUMN {name} {decl}")
    conn.commit()


def update_geometry(conn: sqlite3.Connection, geometries: list[Any]) -> None:
    """UPDATE detections SET <geometry cols> WHERE detection_id=? for each BoxGeometry."""
    if not geometries:
        return
    assignments = ", ".join(f"{name}=?" for name in _GEOMETRY_COLUMNS)
    rows = [
        tuple(getattr(g, name) for name in _GEOMETRY_COLUMNS) + (int(g.detection_id),)
        for g in geometries
    ]
    conn.executemany(f"UPDATE detections SET {assignments} WHERE detection_id=?", rows)
    conn.commit()


def read_detection_geometry(conn: sqlite3.Connection, detection_id: int) -> dict[str, Any]:
    """Read geometry for one detection, applying defaults for missing/NULL columns so
    pre-feature tables (no geometry columns) remain readable (Requirement 1.11)."""
    cols = _existing_columns(conn, "detections")
    row = conn.execute(
        "SELECT * FROM detections WHERE detection_id = ?", (int(detection_id),)
    ).fetchone()
    if row is None:
        raise KeyError(detection_id)
    out: dict[str, Any] = {}
    for name, default in _GEOMETRY_DEFAULTS.items():
        value = row[name] if name in cols and row[name] is not None else default
        out[name] = value
    return out


# --- C2B: reorder path (additive gdino_* columns + deterministic label fallback) ------

import enum  # noqa: E402

_GDINO_DDL: tuple[tuple[str, str], ...] = (
    ("gdino_label", "TEXT"),
    ("gdino_score", "REAL"),
    ("gdino_prompt_key", "TEXT"),
    ("predicted_label", "TEXT"),  # NULL until step2 commits a label (reorder path)
)


class PipelineOrder(enum.Enum):
    LABEL_FIRST = "label_first"   # default: step2 labels, then step4 dedups
    DEDUP_FIRST = "dedup_first"   # reorder: step4 dedups, then step2 labels the kept set


def ensure_gdino_columns(conn: sqlite3.Connection) -> None:
    """Idempotent additive migration for the reorder path. Never drops columns."""
    existing = _existing_columns(conn, "detections")
    for name, decl in _GDINO_DDL:
        if name not in existing:
            conn.execute(f"ALTER TABLE detections ADD COLUMN {name} {decl}")
    conn.commit()


def backfill_gdino_columns(conn: sqlite3.Connection) -> None:
    """Forward migration: populate gdino_* from existing label/score/prompt_key.
    Idempotent (only fills NULLs), non-destructive."""
    ensure_gdino_columns(conn)
    conn.execute(
        """
        UPDATE detections
           SET gdino_label      = COALESCE(gdino_label, label),
               gdino_score      = COALESCE(gdino_score, score),
               gdino_prompt_key = COALESCE(gdino_prompt_key, prompt_key)
         WHERE gdino_label IS NULL OR gdino_score IS NULL OR gdino_prompt_key IS NULL
        """
    )
    conn.commit()


def read_predicted_label(row: Any) -> str:
    """Deterministic fallback chain; never returns NULL/empty under the reorder path:
    committed predicted_label -> gdino_label -> legacy label column."""
    keys = row.keys() if hasattr(row, "keys") else row
    for name in ("predicted_label", "gdino_label", "label"):
        if name in keys:
            value = row[name]
            if value:
                return str(value)
    return ""


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
            ensure_geometry_columns(self.conn)
            self.conn.commit()

    def update_geometry(self, geometries: list[Any]) -> None:
        with self._lock:
            update_geometry(self.conn, geometries)

    def fetch_image_boxes(self, *, run_id: str, image_id: int) -> list[tuple]:
        """Return (detection_id, x1, y1, x2, y2, label) for every detection of one image."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT detection_id, x1, y1, x2, y2, label FROM detections"
                " WHERE run_id = ? AND image_id = ? ORDER BY detection_id",
                (run_id, int(image_id)),
            ).fetchall()
        return [
            (int(r["detection_id"]), float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]), str(r["label"]))
            for r in rows
        ]

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

    def upsert_image(self, *, run_id: str, image: ImageInfo, stored_path: str, status: str = "pending") -> int:
        with self._lock:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO images (run_id, rel_path, path, name, width, height, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, image.rel_path, str(stored_path), image.path.name, int(image.width), int(image.height), status),
            )
            self.conn.execute(
                """
                UPDATE images
                SET path = ?, name = ?, width = ?, height = ?, status = ?
                WHERE run_id = ? AND rel_path = ?
                """,
                (str(stored_path), image.path.name, int(image.width), int(image.height), status, run_id, image.rel_path),
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
