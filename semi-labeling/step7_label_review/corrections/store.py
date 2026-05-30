"""C6: correction store (label_corrections). Depends only on stdlib + numpy."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable

import numpy as np

_VALID_TYPES = {"relabel", "confirm", "reject"}
_EMB_DTYPE = "<f4"


class CorrectionWriteError(Exception):
    """Raised when a Label_Correction cannot be persisted; carries the rejected id."""

    def __init__(self, correction_id: str) -> None:
        super().__init__(f"Failed to persist correction: {correction_id}")
        self.correction_id = correction_id


@dataclass(frozen=True)
class LabelCorrection:
    correction_id: str
    created_at_utc: str
    source_run_id: str
    result_id: int
    image_rel_path: str
    original_label: str
    corrected_label: str
    correction_type: str  # 'relabel' | 'confirm' | 'reject'
    reviewer_note: str = ""
    confidence: float = 1.0  # clamped to [0, 1]
    image_hash: str | None = None
    crop_hash: str | None = None
    embedding_blob: bytes | None = None
    export_to_training: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))


_COLUMNS = [f.name for f in fields(LabelCorrection)]


class CorrectionStore:
    def __init__(self, db: str | Path | sqlite3.Connection) -> None:
        if isinstance(db, sqlite3.Connection):
            self.conn = db
        else:
            self.conn = sqlite3.connect(str(Path(db).expanduser()), timeout=60.0)
        self.conn.row_factory = sqlite3.Row
        self.ensure_schema()

    def ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS label_corrections (
                correction_id      TEXT PRIMARY KEY,
                created_at_utc     TEXT NOT NULL,
                source_run_id      TEXT NOT NULL,
                result_id          INTEGER NOT NULL,
                image_rel_path     TEXT NOT NULL,
                original_label     TEXT NOT NULL,
                corrected_label    TEXT NOT NULL,
                correction_type    TEXT NOT NULL,
                reviewer_note      TEXT DEFAULT '',
                confidence         REAL DEFAULT 1.0,
                image_hash         TEXT,
                crop_hash          TEXT,
                embedding_blob     BLOB,
                export_to_training INTEGER DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_corrections_label
            ON label_corrections(original_label, corrected_label);
            CREATE INDEX IF NOT EXISTS idx_corrections_image
            ON label_corrections(image_rel_path);
            """
        )
        self.conn.commit()

    def add_correction(self, correction: LabelCorrection) -> None:
        if correction.correction_type not in _VALID_TYPES:
            raise ValueError(f"Invalid correction_type: {correction.correction_type!r}")
        placeholders = ", ".join("?" for _ in _COLUMNS)
        values = tuple(getattr(correction, name) for name in _COLUMNS)
        try:
            self.conn.execute(
                f"INSERT INTO label_corrections ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
                values,
            )
            self.conn.commit()
        except Exception as exc:  # noqa: BLE001 - transactional: leave prior rows intact
            self.conn.rollback()
            raise CorrectionWriteError(correction.correction_id) from exc

    def iter_corrections(self) -> Iterable[LabelCorrection]:
        for row in self.conn.execute("SELECT * FROM label_corrections ORDER BY correction_id"):
            yield LabelCorrection(**{name: row[name] for name in _COLUMNS})

    def correction_embeddings(self) -> tuple[np.ndarray, list[str]]:
        """(matrix, original_labels) for rows with an embedding_blob. Returns an empty
        matrix shape (0, 0) when no correction carries an embedding (autoflag no-op signal)."""
        vectors: list[np.ndarray] = []
        labels: list[str] = []
        for row in self.conn.execute(
            "SELECT original_label, embedding_blob FROM label_corrections WHERE embedding_blob IS NOT NULL ORDER BY correction_id"
        ):
            vectors.append(np.frombuffer(row["embedding_blob"], dtype=_EMB_DTYPE).astype(np.float32))
            labels.append(str(row["original_label"]))
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32), []
        return np.vstack(vectors), labels
