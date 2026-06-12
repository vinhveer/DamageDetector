from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class SemanticRunMetadata:
    semantic_run_id: str
    created_at_utc: str
    source_run_id: str
    source_stage: str
    model_name: str
    pretrained: str
    device: str
    prompt_config_json: str
    options_json: str


@dataclass(frozen=True)
class SourceDetection:
    result_id: int
    source_detection_id: int
    image_id: int
    image_rel_path: str
    image_path: str
    source_input_dir: str
    image_width: int
    image_height: int
    prompt_key: str
    detector_label: str
    detector_score: float
    x1: float
    y1: float
    x2: float
    y2: float
    crop_path: str
    initial_label: str
    initial_probability: float
    scores: dict[str, float] = field(default_factory=dict)


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).expanduser().resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def resolve_dedup_run_id(conn: sqlite3.Connection, requested: str) -> str:
    raw = str(requested or "latest").strip()
    if raw and raw.lower() != "latest":
        row = conn.execute("SELECT dedup_run_id FROM dedup_runs WHERE dedup_run_id = ?", (raw,)).fetchone()
        if row is None:
            raise RuntimeError(f"Dedup run not found: {raw}")
        return raw
    row = conn.execute(
        "SELECT dedup_run_id FROM dedup_runs ORDER BY created_at_utc DESC, dedup_run_id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("No dedup run found in dedup DB.")
    return str(row["dedup_run_id"])


def read_kept_result_ids(conn: sqlite3.Connection, *, dedup_run_id: str) -> set[int]:
    rows = conn.execute(
        "SELECT result_id FROM dedup_results WHERE dedup_run_id = ? AND keep = 1 ORDER BY result_id",
        (dedup_run_id,),
    ).fetchall()
    return {int(row["result_id"]) for row in rows}


# ---------------------------------------------------------------------------
# GDINO-only source reader (OpenCLIP removed)
# ---------------------------------------------------------------------------
#
# The pipeline no longer runs OpenCLIP.  Step 01 reads GroundingDINO detections
# directly from the damage_scan tables (``runs`` / ``images`` / ``detections``)
# that the ``detect`` step writes into pipeline.sqlite3.  The returned
# ``SourceDetection`` has an empty ``scores`` map, so downstream code (semantic
# ensemble + decision policy) falls back to the detector seed label.

_DAMAGE_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "crack": ("crack", "fracture", "fissure"),
    "spall": ("spall", "spalling", "delamination", "flaking", "broken", "chipped"),
    "mold": ("mold", "mould", "mildew", "moss", "stain"),
}


def _normalise_damage_label(detector_label: str, prompt_key: str) -> str:
    haystack = f"{detector_label} {prompt_key}".lower()
    for label, words in _DAMAGE_LABEL_ALIASES.items():
        if any(word in haystack for word in words):
            return label
    return str(detector_label or "").strip().lower()


def resolve_source_run_id(conn: sqlite3.Connection, requested: str) -> str:
    """Resolve a damage_scan run id (or 'latest')."""
    raw = str(requested or "latest").strip()
    if raw and raw.lower() != "latest":
        row = conn.execute("SELECT run_id FROM runs WHERE run_id = ?", (raw,)).fetchone()
        if row is None:
            raise RuntimeError(f"Source run not found: {raw}")
        return raw
    row = conn.execute(
        "SELECT run_id FROM runs ORDER BY created_at_utc DESC, run_id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("No damage_scan run found in source DB.")
    return str(row["run_id"])


def read_source_run_metadata(conn: sqlite3.Connection, source_run_id: str, *, stage: str = "final") -> SemanticRunMetadata:
    row = conn.execute(
        "SELECT run_id, created_at_utc, detector_name, checkpoint, device FROM runs WHERE run_id = ?",
        (source_run_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Source run metadata not found: {source_run_id}")
    return SemanticRunMetadata(
        semantic_run_id=str(row["run_id"]),
        created_at_utc=str(row["created_at_utc"]),
        source_run_id=str(row["run_id"]),
        source_stage=str(stage),
        model_name=str(row["detector_name"] or "groundingdino"),
        pretrained=str(row["checkpoint"] or ""),
        device=str(row["device"] or ""),
        prompt_config_json="{}",
        options_json="{}",
    )


def read_gdino_detections(
    conn: sqlite3.Connection,
    *,
    source_run_id: str,
    stage: str = "final",
    labels: Iterable[str] | None = None,
    kept_result_ids: set[int] | None = None,
    limit: int = 0,
) -> list[SourceDetection]:
    label_list = [str(item).strip() for item in (labels or []) if str(item).strip()]
    clauses = ["det.run_id = ?", "det.stage = ?"]
    params: list[Any] = [source_run_id, str(stage)]
    if label_list:
        placeholders = ", ".join("?" for _ in label_list)
        clauses.append(f"det.label IN ({placeholders})")
        params.extend(label_list)
    if kept_result_ids is not None:
        if not kept_result_ids:
            return []
        sorted_ids = sorted(int(item) for item in kept_result_ids)
        placeholders = ", ".join("?" for _ in sorted_ids)
        clauses.append(f"det.detection_id IN ({placeholders})")
        params.extend(sorted_ids)

    sql = f"""
        SELECT det.detection_id, det.image_id, det.prompt_key, det.label, det.score,
               det.x1, det.y1, det.x2, det.y2,
               img.rel_path AS image_rel_path, img.path AS image_path,
               img.width AS image_width, img.height AS image_height,
               src.input_dir AS source_input_dir
        FROM detections det
        JOIN images img ON img.image_id = det.image_id
        JOIN runs src ON src.run_id = det.run_id
        WHERE {' AND '.join(clauses)}
        ORDER BY det.detection_id
    """
    if int(limit) > 0:
        sql = f"{sql} LIMIT {int(limit)}"
    rows = conn.execute(sql, params).fetchall()
    detections: list[SourceDetection] = []
    for row in rows:
        detector_label = str(row["label"])
        prompt_key = str(row["prompt_key"])
        detections.append(
            SourceDetection(
                result_id=int(row["detection_id"]),
                source_detection_id=int(row["detection_id"]),
                image_id=int(row["image_id"]),
                image_rel_path=str(row["image_rel_path"]),
                image_path=str(row["image_path"]),
                source_input_dir=str(row["source_input_dir"]),
                image_width=int(row["image_width"]),
                image_height=int(row["image_height"]),
                prompt_key=prompt_key,
                detector_label=detector_label,
                detector_score=float(row["score"]),
                x1=float(row["x1"]),
                y1=float(row["y1"]),
                x2=float(row["x2"]),
                y2=float(row["y2"]),
                crop_path="",
                initial_label=_normalise_damage_label(detector_label, prompt_key),
                initial_probability=float(row["score"]),
                scores={},
            )
        )
    return detections
