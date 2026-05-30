from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class Detection:
    result_id: int
    source_detection_id: int
    image_id: int
    image_rel_path: str
    image_path: str
    source_input_dir: str
    predicted_label: str
    detector_label: str
    detector_score: float
    predicted_probability: float
    predicted_probability_pct: float
    x1: float
    y1: float
    x2: float
    y2: float
    image_width: int
    image_height: int
    crop_path: str = ""

    @property
    def label(self) -> str:
        return self.predicted_label

    @property
    def width(self) -> float:
        return max(0.0, float(self.x2) - float(self.x1))

    @property
    def height(self) -> float:
        return max(0.0, float(self.y2) - float(self.y1))

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def image_area(self) -> float:
        return max(1.0, float(self.image_width) * float(self.image_height))

    @property
    def area_ratio(self) -> float:
        return self.area / self.image_area

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (float(self.x1), float(self.y1), float(self.x2), float(self.y2))


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).expanduser().resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def resolve_semantic_run_id(conn: sqlite3.Connection, requested: str) -> str:
    raw = str(requested or "latest").strip()
    if raw and raw.lower() != "latest":
        row = conn.execute(
            "SELECT semantic_run_id FROM openclip_semantic_runs WHERE semantic_run_id = ?",
            (raw,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Semantic run not found: {raw}")
        return raw
    row = conn.execute(
        "SELECT semantic_run_id FROM openclip_semantic_runs ORDER BY created_at_utc DESC, semantic_run_id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("No semantic run found in source DB.")
    return str(row["semantic_run_id"])


def read_detections(
    conn: sqlite3.Connection,
    *,
    semantic_run_id: str,
    labels: Iterable[str] | None = None,
    min_confidence_pct: float = 0.0,
) -> list[Detection]:
    label_list = [str(item).strip() for item in (labels or []) if str(item).strip()]
    clauses = ["res.semantic_run_id = ?", "res.status = 'ok'", "res.predicted_probability_pct >= ?"]
    params: list[Any] = [semantic_run_id, float(min_confidence_pct)]
    if label_list:
        placeholders = ", ".join("?" for _ in label_list)
        clauses.append(f"res.predicted_label IN ({placeholders})")
        params.extend(label_list)

    rows = conn.execute(
        f"""
        SELECT res.result_id, res.source_detection_id, res.image_id,
               res.image_rel_path, res.image_path, runs.input_dir AS source_input_dir,
               res.predicted_label, res.detector_label, res.detector_score,
               res.predicted_probability, res.predicted_probability_pct,
               res.x1, res.y1, res.x2, res.y2, img.width AS image_width, img.height AS image_height,
               res.crop_path
        FROM openclip_semantic_results res
        JOIN images img ON img.image_id = res.image_id
        JOIN runs ON runs.run_id = res.source_run_id
        WHERE {' AND '.join(clauses)}
        ORDER BY res.image_rel_path, res.result_id
        """,
        params,
    ).fetchall()
    return [Detection(**dict(row)) for row in rows]


def limit_to_first_images(detections: list[Detection], limit_images: int) -> list[Detection]:
    if int(limit_images) <= 0:
        return detections
    selected: set[str] = set()
    out: list[Detection] = []
    for detection in detections:
        if detection.image_rel_path not in selected:
            if len(selected) >= int(limit_images):
                break
            selected.add(detection.image_rel_path)
        out.append(detection)
    return out


def groupby_image(detections: Iterable[Detection]) -> dict[str, list[Detection]]:
    groups: dict[str, list[Detection]] = {}
    for detection in detections:
        groups.setdefault(detection.image_rel_path, []).append(detection)
    return groups


def resolve_embedding_run_id(conn: sqlite3.Connection, requested: str) -> tuple[str, int]:
    raw = str(requested or "latest").strip()
    if raw and raw.lower() != "latest":
        row = conn.execute(
            "SELECT embedding_run_id, dim FROM embedding_runs WHERE embedding_run_id = ?",
            (raw,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Embedding run not found: {raw}")
        return str(row["embedding_run_id"]), int(row["dim"])
    row = conn.execute(
        "SELECT embedding_run_id, dim FROM embedding_runs ORDER BY created_at_utc DESC, embedding_run_id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("No embedding run found in embedding DB.")
    return str(row["embedding_run_id"]), int(row["dim"])


def load_embedding_map(
    conn: sqlite3.Connection,
    *,
    embedding_run_id: str,
    dim: int,
    result_ids: Iterable[int],
) -> dict[int, np.ndarray]:
    requested = [int(item) for item in result_ids]
    if not requested:
        return {}

    # C4: when multi-view embeddings exist, dedup reads only the 'tight' view.
    has_view = any(str(row[1]) == "view_name" for row in conn.execute("PRAGMA table_info(detection_embeddings)"))
    view_clause = " AND COALESCE(view_name, 'tight') = 'tight'" if has_view else ""

    out: dict[int, np.ndarray] = {}
    batch_size = 900
    for start in range(0, len(requested), batch_size):
        chunk = requested[start : start + batch_size]
        placeholders = ", ".join("?" for _ in chunk)
        rows = conn.execute(
            f"""
            SELECT result_id, embedding_blob
            FROM detection_embeddings
            WHERE embedding_run_id = ? AND result_id IN ({placeholders}){view_clause}
            """,
            [embedding_run_id, *chunk],
        ).fetchall()
        for row in rows:
            result_id = int(row["result_id"])
            arr = np.frombuffer(row["embedding_blob"], dtype="<f4")
            if arr.size != int(dim):
                raise ValueError(f"Invalid embedding size for result_id={result_id}: got {arr.size}, expected {dim}")
            out[result_id] = arr.astype(np.float32, copy=True)
    return out


def align_embeddings(detections: Iterable[Detection], embedding_map: dict[int, np.ndarray], *, dim: int) -> np.ndarray:
    rows = [embedding_map.get(int(d.result_id), np.zeros((int(dim),), dtype=np.float32)) for d in detections]
    if not rows:
        return np.empty((0, int(dim)), dtype=np.float32)
    return np.vstack(rows).astype(np.float32, copy=False)


def resolve_image_path(detection: Detection, image_root: Path | None) -> Path:
    candidates: list[Path] = []
    rel_path = str(detection.image_rel_path or "").strip()
    stored_path = str(detection.image_path or "").strip()
    source_input_dir = Path(str(detection.source_input_dir or "")).expanduser()

    if image_root is not None:
        root = image_root.expanduser().resolve()
        if rel_path:
            candidates.append(root / rel_path)
        if stored_path:
            candidates.append(root / Path(stored_path).name)

    if stored_path:
        stored = Path(stored_path).expanduser()
        candidates.append(stored if stored.is_absolute() else source_input_dir / stored_path)
    if rel_path:
        candidates.append(source_input_dir / rel_path)
    if stored_path:
        candidates.append(source_input_dir / Path(stored_path).name)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()

    if image_root is not None and rel_path:
        return (image_root.expanduser().resolve() / rel_path).resolve()
    return (source_input_dir / rel_path).expanduser().resolve()
