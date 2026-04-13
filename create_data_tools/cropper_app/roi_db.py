from __future__ import annotations

import sqlite3
from pathlib import Path

from create_data_tools.cropper_app.domain import Roi


class RoiDatabase:
    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    @property
    def path(self) -> Path:
        return self._db_path

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rois (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_rel_path TEXT NOT NULL,
                name TEXT NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                size INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rois_image_rel_path ON rois(image_rel_path);")
        self._conn.commit()

    def list_rois(self, image_rel_path: str) -> list[Roi]:
        cur = self._conn.cursor()
        rows = cur.execute(
            """
            SELECT id, image_rel_path, name, x, y, size
            FROM rois
            WHERE image_rel_path = ?
            ORDER BY id ASC
            """,
            (str(image_rel_path),),
        ).fetchall()
        return [Roi(**dict(row)) for row in rows]

    def list_images_with_rois(self) -> list[str]:
        cur = self._conn.cursor()
        rows = cur.execute("SELECT DISTINCT image_rel_path FROM rois ORDER BY image_rel_path ASC").fetchall()
        return [str(row["image_rel_path"]) for row in rows]

    def create_roi(self, image_rel_path: str, *, name: str, x: int, y: int, size: int) -> Roi:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO rois (image_rel_path, name, x, y, size)
            VALUES (?, ?, ?, ?, ?)
            """,
            (str(image_rel_path), str(name), int(x), int(y), int(size)),
        )
        self._conn.commit()
        roi_id = int(cur.lastrowid)
        return Roi(id=roi_id, image_rel_path=str(image_rel_path), name=str(name), x=int(x), y=int(y), size=int(size))

    def update_roi(self, roi_id: int, *, x: int, y: int, size: int, name: str | None = None) -> None:
        if name is None:
            self._conn.execute(
                "UPDATE rois SET x = ?, y = ?, size = ?, updated_at = datetime('now') WHERE id = ?",
                (int(x), int(y), int(size), int(roi_id)),
            )
        else:
            self._conn.execute(
                "UPDATE rois SET name = ?, x = ?, y = ?, size = ?, updated_at = datetime('now') WHERE id = ?",
                (str(name), int(x), int(y), int(size), int(roi_id)),
            )
        self._conn.commit()

    def delete_roi(self, roi_id: int) -> None:
        self._conn.execute("DELETE FROM rois WHERE id = ?", (int(roi_id),))
        self._conn.commit()

