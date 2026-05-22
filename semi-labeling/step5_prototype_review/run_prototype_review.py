#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

np = None


DEFAULT_PROTOTYPES: dict[str, list[str]] = {
    "crack": ["crack:34", "crack:484"],
    "spall": ["spall:36", "spall:57", "spall:114", "spall:364", "mold:193"],
    "mold": ["mold:676", "mold:51"],
}
LABELS = tuple(DEFAULT_PROTOTYPES.keys())


@dataclass(frozen=True)
class Assignment:
    result_id: int
    source_detection_id: int
    image_id: int
    image_rel_path: str
    image_path: str
    source_input_dir: str
    predicted_label: str
    predicted_probability_pct: float
    detector_score: float
    cluster_key: str
    label_scope: str
    cluster_id: int
    cluster_size: int
    cluster_purity: float
    x1: float
    y1: float
    x2: float
    y2: float
    image_width: int
    image_height: int


@dataclass(frozen=True)
class ClusterScore:
    cluster_key: str
    original_label_scope: str
    original_major_label: str
    cluster_size: int
    purity: float
    recommended_label: str
    crop_vote_label: str
    crop_vote_ratio: float
    mixed_ratio: float
    score_by_label: dict[str, float]
    top_score: float
    second_score: float
    confidence_gap: float
    review_bucket: str
    reason: str
    is_prototype: bool


def require_numpy() -> Any:
    global np
    if np is None:
        import numpy as numpy_module

        np = numpy_module
    return np


def l2_normalize(arr: Any) -> Any:
    numpy_module = require_numpy()
    values = numpy_module.asarray(arr, dtype=numpy_module.float32)
    norms = numpy_module.linalg.norm(values, axis=1, keepdims=True)
    norms = numpy_module.maximum(norms, 1e-12)
    return values / norms


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "object_detection").exists() and (candidate / "tools").exists():
            return candidate
    return current.parents[2]


REPO_ROOT = repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def default_source_db() -> Path:
    return REPO_ROOT.parent / "infer_results" / "semi-labeling" / "step2_sematic" / "damage_scan.sqlite3"


def default_feature_db() -> Path:
    return REPO_ROOT.parent / "infer_results" / "semi-labeling" / "step4_feature_grouping" / "feature_groups.sqlite3"


def default_output_dir() -> Path:
    return REPO_ROOT.parent / "infer_results" / "semi-labeling" / "step5_prototype_review"


def default_image_root() -> Path:
    return REPO_ROOT.parent / "HinhAnh"


def connect_ro(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path.expanduser().resolve()}?mode=ro", uri=True, timeout=60.0)
    conn.row_factory = sqlite3.Row
    return conn


def connect_rw(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def latest_grouping_run_id(feature_db: Path) -> str:
    conn = connect_ro(feature_db)
    try:
        row = conn.execute(
            "SELECT grouping_run_id FROM feature_group_runs ORDER BY created_at_utc DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise RuntimeError(f"No grouping run found in {feature_db}")
    return str(row["grouping_run_id"])


def load_run(feature_db: Path, grouping_run_id: str) -> dict[str, Any]:
    conn = connect_ro(feature_db)
    try:
        row = conn.execute(
            """
            SELECT grouping_run_id, source_db_path, filtered_db_path, source_filter_run_id,
                   model_name, device, options_json, total_boxes, total_clusters
            FROM feature_group_runs
            WHERE grouping_run_id = ?
            """,
            (grouping_run_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise RuntimeError(f"grouping_run_id not found: {grouping_run_id}")
    return dict(row)


def load_assignments(feature_db: Path, source_db: Path, grouping_run_id: str) -> list[Assignment]:
    conn = connect_ro(feature_db)
    conn.execute("ATTACH DATABASE ? AS src", (str(source_db.expanduser().resolve()),))
    try:
        rows = conn.execute(
            """
            SELECT a.result_id, a.source_detection_id, res.image_id, a.image_rel_path,
                   res.image_path, src_run.input_dir AS source_input_dir,
                   a.predicted_label, a.predicted_probability_pct, a.detector_score,
                   a.cluster_key, a.label_scope, a.cluster_id, a.cluster_size,
                   a.cluster_purity, res.x1, res.y1, res.x2, res.y2,
                   img.width AS image_width, img.height AS image_height
            FROM feature_group_assignments a
            JOIN src.openclip_semantic_results res ON res.result_id = a.result_id
            JOIN src.images img ON img.image_id = res.image_id
            JOIN src.runs src_run ON src_run.run_id = res.source_run_id
            WHERE a.grouping_run_id = ?
            ORDER BY a.cluster_key, a.distance_to_center, a.result_id
            """,
            (grouping_run_id,),
        ).fetchall()
    finally:
        conn.close()
    return [Assignment(**dict(row)) for row in rows]


def load_cluster_summaries(feature_db: Path, grouping_run_id: str) -> dict[str, dict[str, Any]]:
    conn = connect_ro(feature_db)
    try:
        rows = conn.execute(
            """
            SELECT cluster_key, predicted_label_scope, cluster_size, major_label, purity,
                   crack_count, mold_count, spall_count, outlier_count
            FROM feature_group_clusters
            WHERE grouping_run_id = ?
            """,
            (grouping_run_id,),
        ).fetchall()
    finally:
        conn.close()
    return {str(row["cluster_key"]): dict(row) for row in rows}


def _empty_proto_entry() -> dict[str, list]:
    """Return a fresh empty prototype entry with clusters and images lists."""
    return {"clusters": [], "images": []}


def parse_prototypes(raw_json: str) -> dict[str, dict[str, list]]:
    """Parse prototype JSON supporting both legacy and v2 formats.

    Legacy format: {"crack": ["c#0001"], "spall": [...], "mold": [...]}
    V2 format:     {"crack": {"clusters": [...], "images": [...]}, ..., "excluded": {...}}

    Returns normalized struct: dict[str, dict[str, list]] with keys `clusters`
    and `images` per label, plus an `excluded` key.
    """
    VALID_KEYS = set(LABELS) | {"excluded"}

    if not str(raw_json or "").strip():
        out: dict[str, dict[str, list]] = {
            label: {"clusters": list(keys), "images": []}
            for label, keys in DEFAULT_PROTOTYPES.items()
        }
        out["excluded"] = _empty_proto_entry()
        return out

    payload = json.loads(raw_json)
    out = {}
    for label, value in payload.items():
        normalized = str(label).strip().lower()
        if normalized not in VALID_KEYS:
            raise ValueError(f"Unsupported prototype label: {label}")
        if isinstance(value, dict):
            # v2 format: value has clusters/images keys
            clusters_raw = value.get("clusters", [])
            images_raw = value.get("images", [])
            out[normalized] = {
                "clusters": [str(item).strip() for item in clusters_raw if str(item).strip()],
                "images": [int(item) for item in images_raw],
            }
        else:
            # legacy format: value is a list of cluster keys
            out[normalized] = {
                "clusters": [str(item).strip() for item in value if str(item).strip()],
                "images": [],
            }

    # Ensure all required labels exist with defaults
    for label in LABELS:
        out.setdefault(label, _empty_proto_entry())
    # Ensure excluded key exists
    out.setdefault("excluded", _empty_proto_entry())
    return out


class DinoV2Embedder:
    def __init__(self, *, model_name: str, device: str) -> None:
        from torch_runtime import describe_device_fallback, select_device_str
        from transformers import AutoImageProcessor, AutoModel

        self.device = select_device_str(device)
        fallback = describe_device_fallback(device, self.device)
        if fallback:
            print(fallback, flush=True)
        local_files_only = Path(model_name).expanduser().exists()
        self.processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.model.to(self.device)
        self.model.eval()
        self.model_name = model_name

    def embed(self, images: list[Image.Image], *, batch_size: int) -> np.ndarray:
        numpy_module = require_numpy()
        import torch

        rows = []
        effective_batch_size = max(1, int(batch_size))
        for start in range(0, len(images), effective_batch_size):
            batch = images[start : start + effective_batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = self.model(**inputs)
                tokens = getattr(outputs, "last_hidden_state", None)
                if tokens is None:
                    raise RuntimeError("DINOv2 model did not return last_hidden_state.")
                pooled = tokens[:, 0]
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            rows.append(pooled.detach().cpu().numpy().astype(numpy_module.float32))
        arr = numpy_module.vstack(rows) if rows else numpy_module.empty((0, 0), dtype=numpy_module.float32)
        return l2_normalize(arr) if arr.size else arr


def resolve_image_path(row: Assignment, image_root: Path | None) -> Path:
    candidates: list[Path] = []
    rel_path = str(row.image_rel_path or "").strip()
    stored_path = str(row.image_path or "").strip()
    source_input_dir = Path(str(row.source_input_dir or "")).expanduser()
    if image_root is not None:
        root = image_root.expanduser().resolve()
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
    if image_root is not None:
        return (image_root.expanduser().resolve() / rel_path).resolve()
    return (source_input_dir / rel_path).expanduser().resolve()


def crop_assignment(row: Assignment, image_root: Path | None, *, padding_ratio: float) -> Image.Image:
    from PIL import Image

    image_path = resolve_image_path(row, image_root)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        pad_x = max(0.0, float(row.x2) - float(row.x1)) * float(padding_ratio)
        pad_y = max(0.0, float(row.y2) - float(row.y1)) * float(padding_ratio)
        x1 = max(0, int(math.floor(float(row.x1) - pad_x)))
        y1 = max(0, int(math.floor(float(row.y1) - pad_y)))
        x2 = min(width, int(math.ceil(float(row.x2) + pad_x)))
        y2 = min(height, int(math.ceil(float(row.y2) + pad_y)))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid crop box result_id={row.result_id}: {(x1, y1, x2, y2)}")
        return rgb.crop((x1, y1, x2, y2))


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS prototype_review_runs (
            review_run_id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            grouping_run_id TEXT NOT NULL,
            source_db_path TEXT NOT NULL,
            feature_db_path TEXT NOT NULL,
            model_name TEXT NOT NULL,
            device TEXT NOT NULL,
            prototype_config_json TEXT NOT NULL,
            thresholds_json TEXT NOT NULL,
            total_clusters INTEGER NOT NULL,
            auto_accept_clusters INTEGER NOT NULL,
            need_review_clusters INTEGER NOT NULL,
            parent_review_run_id TEXT,
            display_name TEXT,
            is_active INTEGER NOT NULL DEFAULT 0,
            is_archived INTEGER NOT NULL DEFAULT 0,
            excluded_image_ids_json TEXT
        );
        CREATE TABLE IF NOT EXISTS prototype_groups (
            review_run_id TEXT NOT NULL,
            target_label TEXT NOT NULL,
            cluster_key TEXT NOT NULL,
            source_label_scope TEXT NOT NULL,
            rows_count INTEGER NOT NULL,
            images_count INTEGER NOT NULL,
            PRIMARY KEY (review_run_id, target_label, cluster_key)
        );
        CREATE TABLE IF NOT EXISTS prototype_embeddings (
            model_name TEXT NOT NULL,
            result_id INTEGER NOT NULL,
            dim INTEGER NOT NULL,
            embedding_blob BLOB NOT NULL,
            PRIMARY KEY (model_name, result_id)
        );
        CREATE TABLE IF NOT EXISTS prototype_cluster_scores (
            review_run_id TEXT NOT NULL,
            cluster_key TEXT NOT NULL,
            original_label_scope TEXT NOT NULL,
            original_major_label TEXT NOT NULL,
            cluster_size INTEGER NOT NULL,
            purity REAL NOT NULL,
            recommended_label TEXT NOT NULL,
            crop_vote_label TEXT NOT NULL,
            crop_vote_ratio REAL NOT NULL,
            mixed_ratio REAL NOT NULL,
            score_crack REAL NOT NULL,
            score_spall REAL NOT NULL,
            score_mold REAL NOT NULL,
            top_score REAL NOT NULL,
            second_score REAL NOT NULL,
            confidence_gap REAL NOT NULL,
            review_bucket TEXT NOT NULL,
            reason TEXT NOT NULL,
            is_prototype INTEGER NOT NULL,
            PRIMARY KEY (review_run_id, cluster_key)
        );
        CREATE TABLE IF NOT EXISTS prototype_assignment_votes (
            review_run_id TEXT NOT NULL,
            result_id INTEGER NOT NULL,
            cluster_key TEXT NOT NULL,
            vote_label TEXT NOT NULL,
            vote_score REAL NOT NULL,
            is_excluded INTEGER DEFAULT 0,
            PRIMARY KEY (review_run_id, result_id)
        );
        CREATE INDEX IF NOT EXISTS idx_prototype_scores_bucket ON prototype_cluster_scores (review_run_id, review_bucket, top_score DESC);
        CREATE INDEX IF NOT EXISTS idx_prototype_votes_cluster ON prototype_assignment_votes (review_run_id, cluster_key);
        """
    )
    conn.commit()
    _migrate_versioning(conn)


def _migrate_votes_is_excluded(conn: sqlite3.Connection) -> None:
    """Idempotent migration: add is_excluded column to prototype_assignment_votes if missing."""
    vote_cols = {row[1] for row in conn.execute("PRAGMA table_info(prototype_assignment_votes)").fetchall()}
    if "is_excluded" not in vote_cols:
        conn.execute("ALTER TABLE prototype_assignment_votes ADD COLUMN is_excluded INTEGER DEFAULT 0")
        conn.commit()


def _migrate_versioning(conn: sqlite3.Connection) -> None:
    """Idempotent migration: add versioning columns to prototype_review_runs if missing."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(prototype_review_runs)").fetchall()}
    needed = {"parent_review_run_id", "display_name", "is_active", "is_archived", "excluded_image_ids_json"}
    if needed.issubset(cols):
        # All prototype_review_runs columns exist — migrate votes table then ensure indexes
        _migrate_votes_is_excluded(conn)
        conn.executescript("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_prototype_run_active
              ON prototype_review_runs (grouping_run_id) WHERE is_active = 1 AND is_archived = 0;
            CREATE INDEX IF NOT EXISTS idx_prototype_run_parent
              ON prototype_review_runs (parent_review_run_id);
        """)
        conn.commit()
        return
    # Add missing columns
    for col, ddl in [
        ("parent_review_run_id", "TEXT"),
        ("display_name", "TEXT"),
        ("is_active", "INTEGER NOT NULL DEFAULT 0"),
        ("is_archived", "INTEGER NOT NULL DEFAULT 0"),
        ("excluded_image_ids_json", "TEXT"),
    ]:
        if col not in cols:
            conn.execute(f"ALTER TABLE prototype_review_runs ADD COLUMN {col} {ddl}")
    conn.commit()
    # Backfill: assign display_name and is_active for existing rows
    rows = conn.execute(
        "SELECT review_run_id, grouping_run_id, created_at_utc FROM prototype_review_runs ORDER BY grouping_run_id, created_at_utc"
    ).fetchall()
    groups: dict[str, list[str]] = {}
    for rid, gid, _ in rows:
        groups.setdefault(gid, []).append(rid)
    for gid, rids in groups.items():
        for idx, rid in enumerate(rids, 1):
            conn.execute("UPDATE prototype_review_runs SET display_name = ? WHERE review_run_id = ?", (f"v{idx}", rid))
        # Latest = active
        conn.execute("UPDATE prototype_review_runs SET is_active = 1 WHERE review_run_id = ?", (rids[-1],))
    conn.commit()
    _migrate_votes_is_excluded(conn)
    conn.executescript("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_prototype_run_active
          ON prototype_review_runs (grouping_run_id) WHERE is_active = 1 AND is_archived = 0;
        CREATE INDEX IF NOT EXISTS idx_prototype_run_parent
          ON prototype_review_runs (parent_review_run_id);
    """)
    conn.commit()
    # Checkpoint WAL so Node.js read-only connections don't hit stale SHM
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        pass


def load_cached_embeddings(conn: sqlite3.Connection, *, model_name: str, result_ids: list[int]) -> dict[int, np.ndarray]:
    numpy_module = require_numpy()
    if not result_ids:
        return {}
    out: dict[int, np.ndarray] = {}
    chunk_size = 900
    for start in range(0, len(result_ids), chunk_size):
        chunk = result_ids[start : start + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"""
            SELECT result_id, dim, embedding_blob
            FROM prototype_embeddings
            WHERE model_name = ? AND result_id IN ({placeholders})
            """,
            (model_name, *chunk),
        ).fetchall()
        for row in rows:
            arr = numpy_module.frombuffer(row["embedding_blob"], dtype=numpy_module.float32).copy()
            dim = int(row["dim"])
            if arr.shape[0] == dim:
                out[int(row["result_id"])] = arr
    return out


def cache_embeddings(conn: sqlite3.Connection, *, model_name: str, rows: list[Assignment], embeddings: np.ndarray) -> None:
    numpy_module = require_numpy()
    if not rows:
        return
    payload = [
        (model_name, int(row.result_id), int(embeddings[idx].shape[0]), embeddings[idx].astype(numpy_module.float32).tobytes())
        for idx, row in enumerate(rows)
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO prototype_embeddings (model_name, result_id, dim, embedding_blob)
        VALUES (?, ?, ?, ?)
        """,
        payload,
    )
    conn.commit()


def embed_missing(
    conn: sqlite3.Connection,
    *,
    model_name: str,
    device: str,
    rows: list[Assignment],
    cached: dict[int, np.ndarray],
    image_root: Path | None,
    batch_size: int,
    padding_ratio: float,
    log_every: int,
) -> dict[int, np.ndarray]:
    missing = [row for row in rows if int(row.result_id) not in cached]
    if not missing:
        return cached
    print(f"embedding missing crops={len(missing)} cached={len(cached)} model={model_name}", flush=True)
    embedder = DinoV2Embedder(model_name=model_name, device=device)
    for start in range(0, len(missing), max(1, int(batch_size))):
        end = min(len(missing), start + max(1, int(batch_size)))
        if log_every > 0 and (start == 0 or start % log_every == 0):
            print(f"embedding crops {start + 1}-{end}/{len(missing)}", flush=True)
        batch_rows = missing[start:end]
        crops = [crop_assignment(row, image_root, padding_ratio=padding_ratio) for row in batch_rows]
        try:
            embeddings = embedder.embed(crops, batch_size=batch_size)
        finally:
            for crop in crops:
                crop.close()
        cache_embeddings(conn, model_name=model_name, rows=batch_rows, embeddings=embeddings)
        for idx, row in enumerate(batch_rows):
            cached[int(row.result_id)] = embeddings[idx]
    return cached


def normalized_mean(vectors: list[np.ndarray]) -> np.ndarray:
    numpy_module = require_numpy()
    if not vectors:
        raise ValueError("Cannot average empty vector list")
    arr = numpy_module.vstack(vectors).astype(numpy_module.float32)
    return l2_normalize(arr.mean(axis=0, keepdims=True))[0].astype(numpy_module.float32)


def group_rows(assignments: list[Assignment]) -> dict[str, list[Assignment]]:
    grouped: dict[str, list[Assignment]] = {}
    for row in assignments:
        grouped.setdefault(row.cluster_key, []).append(row)
    return grouped


def build_prototype_centroids(
    grouped: dict[str, list[Assignment]],
    embeddings: dict[int, np.ndarray],
    prototypes: dict[str, list[str]],
    image_picks: dict[str, list[int]] | None = None,
) -> tuple[dict[str, list[tuple[str, np.ndarray]]], set[str], set[int]]:
    centroids: dict[str, list[tuple[str, np.ndarray]]] = {label: [] for label in LABELS}
    prototype_keys: set[str] = set()
    prototype_image_ids: set[int] = set()
    # Cluster-based centroids
    for label, keys in prototypes.items():
        for key in keys:
            rows = grouped.get(key, [])
            if not rows:
                raise RuntimeError(f"Prototype cluster not found: {key}")
            vectors = [embeddings[int(row.result_id)] for row in rows]
            centroids[label].append((key, normalized_mean(vectors)))
            prototype_keys.add(key)
    # Image-based centroids (single-image picks)
    if image_picks:
        for label, rids in image_picks.items():
            for rid in rids:
                if rid not in embeddings:
                    raise RuntimeError(f"Embedding missing for image {rid}")
                centroids[label].append((f"img#{rid}", embeddings[rid]))
                prototype_image_ids.add(rid)
    # Validation: each label must have at least one centroid (cluster or image)
    for label in LABELS:
        if not centroids[label]:
            raise RuntimeError(f"No prototype centroids for label={label}")
    return centroids, prototype_keys, prototype_image_ids


def label_scores(vector: np.ndarray, prototype_centroids: dict[str, list[tuple[str, np.ndarray]]]) -> dict[str, float]:
    numpy_module = require_numpy()
    scores: dict[str, float] = {}
    for label, centroids in prototype_centroids.items():
        scores[label] = max(float(numpy_module.dot(vector, centroid)) for _key, centroid in centroids)
    return scores


def sorted_scores(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def bucket_score(
    *,
    cluster_key: str,
    excluded_keys: set[str],
    recommended_label: str,
    original_label: str,
    top_score: float,
    confidence_gap: float,
    crop_vote_ratio: float,
    is_prototype: bool,
    unknown_threshold: float,
    auto_threshold: float,
    gap_threshold: float,
    vote_threshold: float,
) -> tuple[str, str]:
    if cluster_key in excluded_keys:
        return "excluded", "manually excluded"
    if is_prototype:
        return "prototype", "selected prototype group"
    if top_score < unknown_threshold:
        return "unknown", "low prototype similarity"
    if confidence_gap < gap_threshold:
        return "ambiguous", "top labels are too close"
    if crop_vote_ratio < vote_threshold:
        return "mixed", "crop votes are split"
    if recommended_label != str(original_label or "").lower():
        return "label_conflict", "prototype label differs from original label"
    if top_score >= auto_threshold:
        return "auto_accept", "high prototype similarity"
    return "need_review", "moderate prototype similarity"


def score_clusters(
    *,
    grouped: dict[str, list[Assignment]],
    summaries: dict[str, dict[str, Any]],
    embeddings: dict[int, np.ndarray],
    prototype_centroids: dict[str, list[tuple[str, np.ndarray]]],
    prototype_keys: set[str],
    excluded_keys: set[str] = frozenset(),
    excluded_ids: set[int] = frozenset(),
    unknown_threshold: float,
    auto_threshold: float,
    gap_threshold: float,
    vote_threshold: float,
) -> tuple[list[ClusterScore], list[tuple[int, str, str, float, int]]]:
    cluster_scores: list[ClusterScore] = []
    assignment_votes: list[tuple[int, str, str, float, int]] = []
    for cluster_key, rows in grouped.items():
        vectors = [embeddings[int(row.result_id)] for row in rows]
        centroid = normalized_mean(vectors)
        scores = label_scores(centroid, prototype_centroids)
        ranked = sorted_scores(scores)
        recommended_label, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence_gap = top_score - second_score

        votes: dict[str, int] = {label: 0 for label in LABELS}
        for row, vector in zip(rows, vectors):
            row_scores = label_scores(vector, prototype_centroids)
            vote_label, vote_score = sorted_scores(row_scores)[0]
            is_excluded = 1 if int(row.result_id) in excluded_ids else 0
            if not is_excluded:
                votes[vote_label] += 1
            assignment_votes.append((int(row.result_id), cluster_key, vote_label, float(vote_score), is_excluded))
        crop_vote_label, crop_vote_count = max(votes.items(), key=lambda item: item[1])
        crop_vote_ratio = float(crop_vote_count) / max(1, len(rows))
        mixed_ratio = 1.0 - crop_vote_ratio

        summary = summaries.get(cluster_key, {})
        original_major_label = str(summary.get("major_label") or rows[0].predicted_label or "").lower()
        is_prototype = cluster_key in prototype_keys
        review_bucket, reason = bucket_score(
            cluster_key=cluster_key,
            excluded_keys=excluded_keys,
            recommended_label=recommended_label,
            original_label=original_major_label,
            top_score=top_score,
            confidence_gap=confidence_gap,
            crop_vote_ratio=crop_vote_ratio,
            is_prototype=is_prototype,
            unknown_threshold=unknown_threshold,
            auto_threshold=auto_threshold,
            gap_threshold=gap_threshold,
            vote_threshold=vote_threshold,
        )
        cluster_scores.append(
            ClusterScore(
                cluster_key=cluster_key,
                original_label_scope=str(summary.get("predicted_label_scope") or rows[0].label_scope),
                original_major_label=original_major_label,
                cluster_size=int(summary.get("cluster_size") or len(rows)),
                purity=float(summary.get("purity") or rows[0].cluster_purity or 0.0),
                recommended_label=recommended_label,
                crop_vote_label=crop_vote_label,
                crop_vote_ratio=crop_vote_ratio,
                mixed_ratio=mixed_ratio,
                score_by_label=scores,
                top_score=top_score,
                second_score=second_score,
                confidence_gap=confidence_gap,
                review_bucket=review_bucket,
                reason=reason,
                is_prototype=is_prototype,
            )
        )
    return cluster_scores, assignment_votes


def write_run(
    conn: sqlite3.Connection,
    *,
    review_run_id: str,
    grouping_run_id: str,
    source_db: Path,
    feature_db: Path,
    model_name: str,
    device: str,
    prototypes: dict[str, list[str]],
    thresholds: dict[str, float],
    grouped: dict[str, list[Assignment]],
    scores: list[ClusterScore],
    votes: list[tuple[int, str, str, float, int]],
    parent_review_run_id: str = "",
    display_name: str = "",
    set_active: bool = False,
    excluded_ids: set[int] | None = None,
) -> None:
    auto_accept = sum(1 for item in scores if item.review_bucket == "auto_accept")
    need_review = sum(1 for item in scores if item.review_bucket not in {"auto_accept", "prototype"})
    # Auto-name if empty
    if not display_name:
        n = conn.execute(
            "SELECT COUNT(*) FROM prototype_review_runs WHERE grouping_run_id = ? AND is_archived = 0",
            (grouping_run_id,),
        ).fetchone()[0]
        display_name = f"v{n + 1}"
    is_active = 1 if set_active else 0
    if set_active:
        conn.execute(
            "UPDATE prototype_review_runs SET is_active = 0 WHERE grouping_run_id = ? AND is_active = 1",
            (grouping_run_id,),
        )
    excluded_image_ids_json = json.dumps(sorted(excluded_ids)) if excluded_ids else None
    conn.execute(
        """
        INSERT INTO prototype_review_runs (
            review_run_id, created_at_utc, grouping_run_id, source_db_path, feature_db_path,
            model_name, device, prototype_config_json, thresholds_json, total_clusters,
            auto_accept_clusters, need_review_clusters,
            parent_review_run_id, display_name, is_active, is_archived, excluded_image_ids_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        """,
        (
            review_run_id,
            datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            grouping_run_id,
            str(source_db),
            str(feature_db),
            model_name,
            device,
            json.dumps(prototypes, ensure_ascii=False, sort_keys=True),
            json.dumps(thresholds, ensure_ascii=False, sort_keys=True),
            len(scores),
            auto_accept,
            need_review,
            parent_review_run_id or None,
            display_name,
            is_active,
            excluded_image_ids_json,
        ),
    )
    prototype_rows = []
    for target_label, keys in prototypes.items():
        for key in keys:
            rows = grouped.get(key, [])
            prototype_rows.append(
                (
                    review_run_id,
                    target_label,
                    key,
                    rows[0].label_scope if rows else "",
                    len(rows),
                    len({row.image_rel_path for row in rows}),
                )
            )
    conn.executemany(
        """
        INSERT INTO prototype_groups (review_run_id, target_label, cluster_key, source_label_scope, rows_count, images_count)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        prototype_rows,
    )
    conn.executemany(
        """
        INSERT INTO prototype_cluster_scores (
            review_run_id, cluster_key, original_label_scope, original_major_label,
            cluster_size, purity, recommended_label, crop_vote_label, crop_vote_ratio,
            mixed_ratio, score_crack, score_spall, score_mold, top_score, second_score,
            confidence_gap, review_bucket, reason, is_prototype
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                review_run_id,
                item.cluster_key,
                item.original_label_scope,
                item.original_major_label,
                item.cluster_size,
                item.purity,
                item.recommended_label,
                item.crop_vote_label,
                item.crop_vote_ratio,
                item.mixed_ratio,
                item.score_by_label.get("crack", 0.0),
                item.score_by_label.get("spall", 0.0),
                item.score_by_label.get("mold", 0.0),
                item.top_score,
                item.second_score,
                item.confidence_gap,
                item.review_bucket,
                item.reason,
                1 if item.is_prototype else 0,
            )
            for item in scores
        ],
    )
    conn.executemany(
        """
        INSERT INTO prototype_assignment_votes (review_run_id, result_id, cluster_key, vote_label, vote_score, is_excluded)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [(review_run_id, result_id, cluster_key, vote_label, vote_score, is_excluded) for result_id, cluster_key, vote_label, vote_score, is_excluded in votes],
    )
    conn.commit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 5 prototype-based review for feature groups.")
    parser.add_argument("--source-db", default=str(default_source_db()))
    parser.add_argument("--feature-db", default=str(default_feature_db()))
    parser.add_argument("--image-root", default=str(default_image_root()))
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    parser.add_argument("--output-db", default="")
    parser.add_argument("--grouping-run-id", default="latest")
    parser.add_argument("--model-name", default="", help="Default: model_name recorded in the Step 4 run.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--padding-ratio", type=float, default=0.05)
    parser.add_argument("--prototype-json", default="", help="JSON object mapping label to cluster keys.")
    parser.add_argument("--unknown-threshold", type=float, default=0.55)
    parser.add_argument("--auto-threshold", type=float, default=0.78)
    parser.add_argument("--gap-threshold", type=float, default=0.03)
    parser.add_argument("--vote-threshold", type=float, default=0.65)
    parser.add_argument("--log-every", type=int, default=256)
    parser.add_argument("--display-name", default="", help="Display name for this version. Auto-generated if empty.")
    parser.add_argument("--parent-review-run-id", default="", help="Lineage pointer to parent version.")
    parser.add_argument("--set-active", action="store_true", help="Set this version as active after insert.")
    parser.add_argument("--archive", action="store_true", help="Standalone mode: archive a review run (requires --target-review-run-id).")
    parser.add_argument("--set-active-only", action="store_true", help="Standalone mode: set active without running review (requires --target-review-run-id).")
    parser.add_argument("--target-review-run-id", default="", help="Target review_run_id for standalone modes.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # ── Standalone modes (no review run) ──────────────────────────────────────
    if args.archive or args.set_active_only:
        target_id = str(args.target_review_run_id or "").strip()
        if not target_id:
            print("ERROR: --target-review-run-id is required for standalone modes", file=sys.stderr)
            return 1
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_db = Path(args.output_db).expanduser().resolve() if str(args.output_db or "").strip() else output_dir / "prototype_review.sqlite3"
        conn = connect_rw(output_db)
        try:
            ensure_schema(conn)
            if args.archive:
                conn.execute("UPDATE prototype_review_runs SET is_archived = 1, is_active = 0 WHERE review_run_id = ?", (target_id,))
                conn.commit()
                print(f"archived review_run_id={target_id}", flush=True)
            else:
                # --set-active-only
                row = conn.execute("SELECT grouping_run_id FROM prototype_review_runs WHERE review_run_id = ?", (target_id,)).fetchone()
                if not row:
                    print(f"ERROR: review_run_id={target_id} not found", file=sys.stderr)
                    return 1
                gid = row[0]
                conn.execute("UPDATE prototype_review_runs SET is_active = 0 WHERE grouping_run_id = ? AND is_active = 1", (gid,))
                conn.execute("UPDATE prototype_review_runs SET is_active = 1 WHERE review_run_id = ?", (target_id,))
                conn.commit()
                print(f"set_active review_run_id={target_id} grouping_run_id={gid}", flush=True)
        finally:
            conn.close()
        return 0

    require_numpy()
    source_db = Path(args.source_db).expanduser().resolve()
    feature_db = Path(args.feature_db).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve() if str(args.image_root or "").strip() else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_db = Path(args.output_db).expanduser().resolve() if str(args.output_db or "").strip() else output_dir / "prototype_review.sqlite3"
    if not source_db.is_file():
        raise FileNotFoundError(f"Source DB not found: {source_db}")
    if not feature_db.is_file():
        raise FileNotFoundError(f"Feature DB not found: {feature_db}")

    grouping_run_id = latest_grouping_run_id(feature_db) if str(args.grouping_run_id).lower() == "latest" else str(args.grouping_run_id)
    run = load_run(feature_db, grouping_run_id)
    model_name = str(args.model_name or run["model_name"])
    parsed = parse_prototypes(args.prototype_json)
    # Split parsed result into cluster keys and image picks for build_prototype_centroids
    prototype_clusters = {label: parsed[label]["clusters"] for label in LABELS}
    image_picks = {label: parsed[label]["images"] for label in LABELS}
    excluded_keys = set(parsed["excluded"]["clusters"])
    excluded_ids = set(parsed["excluded"]["images"])
    thresholds = {
        "unknown_threshold": float(args.unknown_threshold),
        "auto_threshold": float(args.auto_threshold),
        "gap_threshold": float(args.gap_threshold),
        "vote_threshold": float(args.vote_threshold),
    }

    print(f"review grouping_run_id={grouping_run_id} model={model_name}", flush=True)
    assignments = load_assignments(feature_db, source_db, grouping_run_id)
    summaries = load_cluster_summaries(feature_db, grouping_run_id)
    grouped = group_rows(assignments)
    if not grouped:
        raise RuntimeError("No assignments found for Step 5 review.")

    conn = connect_rw(output_db)
    try:
        ensure_schema(conn)
        result_ids = [int(row.result_id) for row in assignments]
        embeddings = load_cached_embeddings(conn, model_name=model_name, result_ids=result_ids)
        embeddings = embed_missing(
            conn,
            model_name=model_name,
            device=str(args.device),
            rows=assignments,
            cached=embeddings,
            image_root=image_root,
            batch_size=int(args.batch_size),
            padding_ratio=float(args.padding_ratio),
            log_every=int(args.log_every),
        )
        prototype_centroids, prototype_keys, prototype_image_ids = build_prototype_centroids(
            grouped, embeddings, prototype_clusters, image_picks=image_picks
        )
        scores, votes = score_clusters(
            grouped=grouped,
            summaries=summaries,
            embeddings=embeddings,
            prototype_centroids=prototype_centroids,
            prototype_keys=prototype_keys,
            excluded_keys=excluded_keys,
            excluded_ids=excluded_ids,
            unknown_threshold=float(args.unknown_threshold),
            auto_threshold=float(args.auto_threshold),
            gap_threshold=float(args.gap_threshold),
            vote_threshold=float(args.vote_threshold),
        )
        review_run_id = uuid.uuid4().hex
        write_run(
            conn,
            review_run_id=review_run_id,
            grouping_run_id=grouping_run_id,
            source_db=source_db,
            feature_db=feature_db,
            model_name=model_name,
            device=str(args.device),
            prototypes=prototype_clusters,
            thresholds=thresholds,
            grouped=grouped,
            scores=scores,
            votes=votes,
            parent_review_run_id=str(args.parent_review_run_id or ""),
            display_name=str(args.display_name or ""),
            set_active=bool(args.set_active),
            excluded_ids=excluded_ids,
        )
    finally:
        conn.close()

    bucket_counts: dict[str, int] = {}
    for item in scores:
        bucket_counts[item.review_bucket] = bucket_counts.get(item.review_bucket, 0) + 1
    print(f"review_run_id={review_run_id}", flush=True)
    print(f"output_db={output_db}", flush=True)
    print(f"clusters={len(scores)} buckets={json.dumps(bucket_counts, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
