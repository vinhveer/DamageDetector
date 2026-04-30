#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

DEFAULT_MODEL_NAME = "facebook/dinov2-large"
LABELS = ("crack", "mold", "spall")


@dataclass(frozen=True)
class SourceBox:
    result_id: int
    source_detection_id: int
    semantic_run_id: str
    source_run_id: str
    image_id: int
    image_rel_path: str
    image_path: str
    source_input_dir: str
    predicted_label: str
    predicted_probability_pct: float
    detector_label: str
    detector_score: float
    x1: float
    y1: float
    x2: float
    y2: float
    image_width: int
    image_height: int
    top_prompt: str

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


@dataclass(frozen=True)
class SpatialPair:
    a_result_id: int
    b_result_id: int
    iou: float
    containment: float
    overlap_a: float
    overlap_b: float


@dataclass(frozen=True)
class FilterDecision:
    result_id: int
    source_detection_id: int
    image_rel_path: str
    predicted_label: str
    keep: bool
    duplicate_group_id: str
    kept_result_id: int
    keep_score: float
    similarity_to_kept: float
    max_similarity: float
    max_iou: float
    max_containment: float
    drop_reason: str
    suspect_reason: str
    area_ratio: float


def resolve_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "object_detection").exists() and (candidate / "tools").exists():
            return candidate
    return current.parents[2]


REPO_ROOT = resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.expanduser().resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=60.0)
    conn.row_factory = sqlite3.Row
    return conn


def connect_output(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def resolve_semantic_run_id(conn: sqlite3.Connection, requested: str) -> str:
    raw = str(requested or "latest").strip()
    if raw and raw.lower() != "latest":
        return raw
    row = conn.execute(
        "SELECT semantic_run_id FROM openclip_semantic_runs ORDER BY created_at_utc DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("No semantic run found in source DB.")
    return str(row["semantic_run_id"])


def parse_labels(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def read_source_boxes(conn: sqlite3.Connection, *, semantic_run_id: str, min_confidence_pct: float, labels: list[str], limit_images: int, image_rel_path: str) -> list[SourceBox]:
    clauses = ["res.semantic_run_id = ?", "res.status = 'ok'", "res.predicted_probability_pct >= ?"]
    params: list[Any] = [semantic_run_id, float(min_confidence_pct)]
    if labels:
        placeholders = ", ".join("?" for _ in labels)
        clauses.append(f"res.predicted_label IN ({placeholders})")
        params.extend(labels)
    image_rel_paths = parse_labels(image_rel_path)
    if image_rel_paths:
        placeholders = ", ".join("?" for _ in image_rel_paths)
        clauses.append(f"res.image_rel_path IN ({placeholders})")
        params.extend(image_rel_paths)
    image_limit_sql = ""
    if int(limit_images) > 0:
        image_limit_sql = """
        AND res.image_rel_path IN (
            SELECT image_rel_path
            FROM openclip_semantic_results
            WHERE semantic_run_id = ? AND status = 'ok'
            GROUP BY image_rel_path
            ORDER BY COUNT(*) DESC, image_rel_path
            LIMIT ?
        )
        """
        params.extend([semantic_run_id, int(limit_images)])
    rows = conn.execute(
        f"""
        SELECT res.result_id, res.source_detection_id, res.semantic_run_id, res.source_run_id,
               res.image_id, res.image_rel_path, res.image_path, src.input_dir AS source_input_dir,
               res.predicted_label, res.predicted_probability_pct, res.detector_label,
               res.detector_score, res.x1, res.y1, res.x2, res.y2, img.width AS image_width,
               img.height AS image_height, res.top_prompt
        FROM openclip_semantic_results res
        JOIN images img ON img.image_id = res.image_id
        JOIN runs src ON src.run_id = res.source_run_id
        WHERE {' AND '.join(clauses)}
        {image_limit_sql}
        ORDER BY res.image_rel_path, res.predicted_label, res.result_id
        """,
        params,
    ).fetchall()
    return [SourceBox(**dict(row)) for row in rows]


def intersection_area(a: SourceBox, b: SourceBox) -> float:
    x1 = max(float(a.x1), float(b.x1))
    y1 = max(float(a.y1), float(b.y1))
    x2 = min(float(a.x2), float(b.x2))
    y2 = min(float(a.y2), float(b.y2))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def spatial_pair(a: SourceBox, b: SourceBox) -> SpatialPair | None:
    inter = intersection_area(a, b)
    if inter <= 0.0:
        return None
    union = max(1e-9, a.area + b.area - inter)
    overlap_a = inter / max(1e-9, a.area)
    overlap_b = inter / max(1e-9, b.area)
    return SpatialPair(a.result_id, b.result_id, inter / union, max(overlap_a, overlap_b), overlap_a, overlap_b)


class UnionFind:
    def __init__(self, ids: Iterable[int]) -> None:
        self.parent = {int(item): int(item) for item in ids}

    def find(self, item: int) -> int:
        item = int(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a

    def groups(self) -> list[list[int]]:
        grouped: dict[int, list[int]] = defaultdict(list)
        for item in self.parent:
            grouped[self.find(item)].append(item)
        return list(grouped.values())


def build_spatial_groups(boxes: list[SourceBox], *, iou_threshold: float, containment_threshold: float) -> tuple[list[list[SourceBox]], dict[tuple[int, int], SpatialPair]]:
    by_id = {box.result_id: box for box in boxes}
    uf = UnionFind(by_id)
    pair_map: dict[tuple[int, int], SpatialPair] = {}
    for i, box_a in enumerate(boxes):
        for box_b in boxes[i + 1 :]:
            pair = spatial_pair(box_a, box_b)
            if pair is None:
                continue
            key = tuple(sorted((box_a.result_id, box_b.result_id)))
            pair_map[key] = pair
            if pair.iou >= iou_threshold or pair.containment >= containment_threshold:
                uf.union(box_a.result_id, box_b.result_id)
    return [[by_id[item] for item in group] for group in uf.groups()], pair_map


def resolve_image_path(box: SourceBox, image_root: Path | None) -> Path:
    candidates: list[Path] = []
    rel_path = str(box.image_rel_path or "").strip()
    stored_path = str(box.image_path or "").strip()
    source_input_dir = Path(str(box.source_input_dir or "")).expanduser()
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


def crop_box(box: SourceBox, image_root: Path | None, *, padding_ratio: float) -> Image.Image:
    image_path = resolve_image_path(box, image_root)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        pad_x = float(box.width) * float(padding_ratio)
        pad_y = float(box.height) * float(padding_ratio)
        x1 = max(0, int(math.floor(float(box.x1) - pad_x)))
        y1 = max(0, int(math.floor(float(box.y1) - pad_y)))
        x2 = min(width, int(math.ceil(float(box.x2) + pad_x)))
        y2 = min(height, int(math.ceil(float(box.y2) + pad_y)))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid crop box result_id={box.result_id}: {(x1, y1, x2, y2)}")
        return rgb.crop((x1, y1, x2, y2))


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

    def embed(self, images: list[Image.Image], *, batch_size: int) -> Any:
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
            rows.append(pooled.detach().cpu())
        return torch.cat(rows, dim=0) if rows else torch.empty((0, 0))


def geometry_score(box: SourceBox) -> float:
    ratio = float(box.area_ratio)
    if ratio <= 0.0:
        return 0.0
    if ratio < 0.001:
        return 0.3
    if ratio <= 0.30:
        return 1.0
    if ratio <= 0.70:
        return 0.5
    return 0.1


def quality_score(box: SourceBox) -> float:
    clip = max(0.0, min(1.0, float(box.predicted_probability_pct) / 100.0))
    detector = max(0.0, min(1.0, float(box.detector_score)))
    return (0.55 * clip) + (0.25 * detector) + (0.20 * geometry_score(box))


def connected_duplicate_components(boxes: list[SourceBox], embeddings: Any, pair_map: dict[tuple[int, int], SpatialPair], *, similarity_threshold: float, contained_similarity_threshold: float, crack_similarity_threshold: float) -> tuple[list[list[int]], dict[tuple[int, int], float]]:
    ids = [int(box.result_id) for box in boxes]
    uf = UnionFind(ids)
    similarity_map: dict[tuple[int, int], float] = {}
    if len(ids) <= 1:
        return [[item] for item in ids], similarity_map
    sims = embeddings @ embeddings.T
    for i, box_a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            box_b = boxes[j]
            key = tuple(sorted((box_a.result_id, box_b.result_id)))
            pair = pair_map.get(key)
            if pair is None:
                continue
            sim = float(sims[i, j])
            similarity_map[key] = sim
            both_crack = box_a.predicted_label.lower() == "crack" and box_b.predicted_label.lower() == "crack"
            if both_crack:
                continue
            threshold = contained_similarity_threshold if pair.containment >= 0.90 else similarity_threshold
            if sim >= threshold:
                uf.union(box_a.result_id, box_b.result_id)
    return uf.groups(), similarity_map


def component_decisions(component: list[SourceBox], *, group_id: str, pair_map: dict[tuple[int, int], SpatialPair], similarity_map: dict[tuple[int, int], float], large_area_threshold: float, large_contains_min: int) -> list[FilterDecision]:
    keep_box = max(component, key=quality_score)
    decisions: list[FilterDecision] = []
    for box in component:
        max_similarity = 0.0
        max_iou = 0.0
        max_containment = 0.0
        similarity_to_kept = 1.0 if box.result_id == keep_box.result_id else 0.0
        for other in component:
            if other.result_id == box.result_id:
                continue
            key = tuple(sorted((box.result_id, other.result_id)))
            pair = pair_map.get(key)
            if pair is not None:
                max_iou = max(max_iou, pair.iou)
                max_containment = max(max_containment, pair.containment)
            sim = similarity_map.get(key, 0.0)
            max_similarity = max(max_similarity, sim)
            if other.result_id == keep_box.result_id:
                similarity_to_kept = sim
        keep = box.result_id == keep_box.result_id
        drop_reason = "" if keep else "feature_duplicate"
        suspect_reason = ""
        contains_children = 0
        for other in component:
            if other.result_id == box.result_id:
                continue
            pair = pair_map.get(tuple(sorted((box.result_id, other.result_id))))
            if pair is not None and box.area > other.area * 1.2 and pair.containment >= 0.90:
                contains_children += 1
        if box.area_ratio >= large_area_threshold and contains_children >= large_contains_min:
            suspect_reason = "large_container_kept" if keep else "large_container_duplicate"
            if not keep:
                drop_reason = "large_container_duplicate"
        decisions.append(
            FilterDecision(
                result_id=box.result_id,
                source_detection_id=box.source_detection_id,
                image_rel_path=box.image_rel_path,
                predicted_label=box.predicted_label,
                keep=keep,
                duplicate_group_id=group_id,
                kept_result_id=keep_box.result_id,
                keep_score=quality_score(box),
                similarity_to_kept=similarity_to_kept,
                max_similarity=max_similarity,
                max_iou=max_iou,
                max_containment=max_containment,
                drop_reason=drop_reason,
                suspect_reason=suspect_reason,
                area_ratio=box.area_ratio,
            )
        )
    return decisions


def singleton_decision(box: SourceBox, *, group_id: str) -> FilterDecision:
    return FilterDecision(
        result_id=box.result_id,
        source_detection_id=box.source_detection_id,
        image_rel_path=box.image_rel_path,
        predicted_label=box.predicted_label,
        keep=True,
        duplicate_group_id=group_id,
        kept_result_id=box.result_id,
        keep_score=quality_score(box),
        similarity_to_kept=1.0,
        max_similarity=0.0,
        max_iou=0.0,
        max_containment=0.0,
        drop_reason="",
        suspect_reason="huge_single_box" if box.area_ratio >= 0.70 else "",
        area_ratio=box.area_ratio,
    )


def suspect_reason(pair: SpatialPair, similarity: float, label_a: str, label_b: str) -> str:
    reasons: list[str] = []
    if pair.iou >= 0.50 and similarity < 0.88:
        reasons.append("high_overlap_low_similarity")
    if similarity >= 0.92 and pair.iou < 0.20 and pair.containment < 0.70:
        reasons.append("high_similarity_low_overlap")
    if label_a != label_b and pair.containment >= 0.70:
        reasons.append("cross_label_overlap")
    return ",".join(reasons)


def apply_multi_feature_container_suppression(
    decisions: list[FilterDecision],
    boxes_by_id: dict[int, SourceBox],
    *,
    containment_threshold: float,
    min_feature_groups: int,
    spatial_only_labels: set[str],
) -> list[FilterDecision]:
    decision_by_id = {item.result_id: item for item in decisions}
    grouped: dict[str, list[FilterDecision]] = defaultdict(list)
    for decision in decisions:
        grouped[decision.image_rel_path].append(decision)

    forced_drop: dict[int, tuple[int, int, float]] = {}
    for image_decisions in grouped.values():
        for parent_decision in image_decisions:
            parent = boxes_by_id[parent_decision.result_id]
            if parent.predicted_label.lower() in spatial_only_labels:
                continue
            child_groups: dict[str, FilterDecision] = {}
            max_containment = 0.0
            for child_decision in image_decisions:
                if child_decision.result_id == parent_decision.result_id:
                    continue
                child = boxes_by_id[child_decision.result_id]
                if parent.area <= child.area * 1.2:
                    continue
                pair = spatial_pair(parent, child)
                if pair is None:
                    continue
                child_inside_parent = pair.overlap_b
                if child_inside_parent < containment_threshold:
                    continue
                max_containment = max(max_containment, child_inside_parent)
                child_groups.setdefault(child_decision.duplicate_group_id, child_decision)
            if len(child_groups) < min_feature_groups:
                continue
            best_child = max(child_groups.values(), key=lambda item: item.keep_score)
            forced_drop[parent_decision.result_id] = (best_child.result_id, len(child_groups), max_containment)

    if not forced_drop:
        return decisions

    out: list[FilterDecision] = []
    for decision in decisions:
        item = forced_drop.get(decision.result_id)
        if item is None:
            out.append(decision)
            continue
        kept_result_id, feature_count, max_containment = item
        previous_suspect = decision.suspect_reason
        suspect_reason = f"multi_feature_container:{feature_count}"
        if previous_suspect:
            suspect_reason = f"{previous_suspect},{suspect_reason}"
        out.append(
            replace(
                decision,
                keep=False,
                kept_result_id=kept_result_id,
                max_containment=max(decision.max_containment, max_containment),
                drop_reason="multi_feature_container",
                suspect_reason=suspect_reason,
            )
        )
    return out


def apply_final_spatial_suppression(
    decisions: list[FilterDecision],
    boxes_by_id: dict[int, SourceBox],
    *,
    group_by_label: bool,
    iou_threshold: float,
    containment_threshold: float,
) -> list[FilterDecision]:
    decision_by_id = {item.result_id: item for item in decisions}
    grouped: dict[tuple[str, str], list[SourceBox]] = defaultdict(list)
    for decision in decisions:
        if not decision.keep:
            continue
        box = boxes_by_id[decision.result_id]
        label_key = box.predicted_label if group_by_label else "all"
        grouped[(box.image_rel_path, label_key)].append(box)

    suppressed_by: dict[int, tuple[int, SpatialPair]] = {}
    for group_boxes in grouped.values():
        selected: list[SourceBox] = []
        for box in sorted(group_boxes, key=quality_score, reverse=True):
            suppressor: tuple[SourceBox, SpatialPair] | None = None
            for kept_box in selected:
                pair = spatial_pair(box, kept_box)
                if pair is None:
                    continue
                if pair.iou >= iou_threshold or pair.containment >= containment_threshold:
                    suppressor = (kept_box, pair)
                    break
            if suppressor is None:
                selected.append(box)
            else:
                kept_box, pair = suppressor
                suppressed_by[box.result_id] = (kept_box.result_id, pair)

    if not suppressed_by:
        return decisions

    out: list[FilterDecision] = []
    for decision in decisions:
        suppressed = suppressed_by.get(decision.result_id)
        if suppressed is None:
            out.append(decision)
            continue
        kept_result_id, pair = suppressed
        previous_reason = decision.suspect_reason
        suspect_reason = "final_spatial_overlap" if not previous_reason else f"{previous_reason},final_spatial_overlap"
        out.append(
            replace(
                decision,
                keep=False,
                kept_result_id=kept_result_id,
                max_iou=max(decision.max_iou, pair.iou),
                max_containment=max(decision.max_containment, pair.containment),
                drop_reason="final_spatial_overlap",
                suspect_reason=suspect_reason,
            )
        )
    return out


def process_boxes(boxes: list[SourceBox], *, embedder: DinoV2Embedder | None, image_root: Path | None, batch_size: int, padding_ratio: float, group_by_label: bool, iou_threshold: float, containment_threshold: float, similarity_threshold: float, contained_similarity_threshold: float, crack_similarity_threshold: float, large_area_threshold: float, large_contains_min: int, spatial_only_labels: set[str], log_every: int) -> tuple[list[FilterDecision], list[tuple[SpatialPair, float, str]]]:
    decisions: list[FilterDecision] = []
    suspect_pairs: list[tuple[SpatialPair, float, str]] = []
    grouped: dict[tuple[str, str], list[SourceBox]] = defaultdict(list)
    for box in boxes:
        label_key = box.predicted_label if group_by_label else "all"
        grouped[(box.image_rel_path, label_key)].append(box)
    total_groups = len(grouped)
    for group_idx, ((image_rel_path, label_key), group_boxes) in enumerate(grouped.items(), start=1):
        if log_every > 0 and (group_idx == 1 or group_idx % log_every == 0):
            print(f"[{group_idx}/{total_groups}] {image_rel_path} label={label_key} boxes={len(group_boxes)}", flush=True)
        spatial_groups, pair_map = build_spatial_groups(group_boxes, iou_threshold=iou_threshold, containment_threshold=containment_threshold)
        if str(label_key).lower() in spatial_only_labels:
            for box in group_boxes:
                decisions.append(singleton_decision(box, group_id=f"{image_rel_path}:{label_key}:{box.result_id}"))
            continue
        if embedder is None:
            raise RuntimeError("DINOv2 embedder is required for non-spatial-only labels.")
        for spatial_group in spatial_groups:
            if len(spatial_group) == 1:
                box = spatial_group[0]
                decisions.append(singleton_decision(box, group_id=f"{image_rel_path}:{label_key}:{box.result_id}"))
                continue
            crops = [crop_box(box, image_root, padding_ratio=padding_ratio) for box in spatial_group]
            embeddings = embedder.embed(crops, batch_size=batch_size)
            components, similarity_map = connected_duplicate_components(
                spatial_group,
                embeddings,
                pair_map,
                similarity_threshold=similarity_threshold,
                contained_similarity_threshold=contained_similarity_threshold,
                crack_similarity_threshold=crack_similarity_threshold,
            )
            box_by_id = {box.result_id: box for box in spatial_group}
            for component_idx, component_ids in enumerate(components, start=1):
                component = [box_by_id[item] for item in component_ids]
                if len(component) == 1:
                    decisions.append(singleton_decision(component[0], group_id=f"{image_rel_path}:{label_key}:{component[0].result_id}"))
                    continue
                group_id = f"{image_rel_path}:{label_key}:{component_idx}:{min(component_ids)}"
                decisions.extend(component_decisions(
                    component,
                    group_id=group_id,
                    pair_map=pair_map,
                    similarity_map=similarity_map,
                    large_area_threshold=large_area_threshold,
                    large_contains_min=large_contains_min,
                ))
            ids_in_group = {box.result_id: box for box in spatial_group}
            for key, pair in pair_map.items():
                if pair.a_result_id not in ids_in_group or pair.b_result_id not in ids_in_group:
                    continue
                sim = similarity_map.get(key, 0.0)
                box_a = ids_in_group[pair.a_result_id]
                box_b = ids_in_group[pair.b_result_id]
                reason = suspect_reason(pair, sim, box_a.predicted_label, box_b.predicted_label)
                if reason:
                    suspect_pairs.append((pair, sim, reason))
    return decisions, suspect_pairs


def ensure_filtered_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS filter_runs (
            filter_run_id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            source_db_path TEXT NOT NULL,
            source_semantic_run_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            device TEXT NOT NULL,
            options_json TEXT NOT NULL,
            total_boxes INTEGER NOT NULL,
            kept_boxes INTEGER NOT NULL,
            dropped_boxes INTEGER NOT NULL,
            suspect_boxes INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS filter_results (
            filter_run_id TEXT NOT NULL,
            result_id INTEGER NOT NULL,
            source_detection_id INTEGER NOT NULL,
            image_rel_path TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            keep INTEGER NOT NULL,
            duplicate_group_id TEXT NOT NULL,
            kept_result_id INTEGER NOT NULL,
            keep_score REAL NOT NULL,
            similarity_to_kept REAL NOT NULL,
            max_similarity REAL NOT NULL,
            max_iou REAL NOT NULL,
            max_containment REAL NOT NULL,
            drop_reason TEXT NOT NULL,
            suspect_reason TEXT NOT NULL,
            area_ratio REAL NOT NULL,
            PRIMARY KEY (filter_run_id, result_id)
        );
        CREATE INDEX IF NOT EXISTS idx_filter_results_image ON filter_results (filter_run_id, image_rel_path, keep);
    """)
    conn.commit()


def ensure_suspect_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS suspect_runs (
            filter_run_id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            source_db_path TEXT NOT NULL,
            source_semantic_run_id TEXT NOT NULL,
            options_json TEXT NOT NULL,
            suspect_pairs INTEGER NOT NULL,
            suspect_boxes INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS suspect_pairs (
            filter_run_id TEXT NOT NULL,
            suspect_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_rel_path TEXT NOT NULL,
            predicted_label_a TEXT NOT NULL,
            predicted_label_b TEXT NOT NULL,
            result_id_a INTEGER NOT NULL,
            result_id_b INTEGER NOT NULL,
            reason TEXT NOT NULL,
            iou REAL NOT NULL,
            containment REAL NOT NULL,
            similarity REAL NOT NULL,
            area_ratio_a REAL NOT NULL,
            area_ratio_b REAL NOT NULL,
            keep_a INTEGER NOT NULL,
            keep_b INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_suspect_pairs_image ON suspect_pairs (filter_run_id, image_rel_path);
    """)
    conn.commit()


def write_filtered_db(db_path: Path, *, filter_run_id: str, source_db_path: Path, semantic_run_id: str, model_name: str, device: str, options: dict[str, Any], decisions: list[FilterDecision]) -> None:
    conn = connect_output(db_path)
    try:
        ensure_filtered_schema(conn)
        kept = sum(1 for item in decisions if item.keep)
        suspect = sum(1 for item in decisions if item.suspect_reason)
        conn.execute(
            """
            INSERT INTO filter_runs (
                filter_run_id, created_at_utc, source_db_path, source_semantic_run_id,
                model_name, device, options_json, total_boxes, kept_boxes, dropped_boxes, suspect_boxes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (filter_run_id, datetime.now(timezone.utc).replace(microsecond=0).isoformat(), str(source_db_path), semantic_run_id, model_name, device, json.dumps(options, ensure_ascii=False, sort_keys=True), len(decisions), kept, len(decisions) - kept, suspect),
        )
        conn.executemany(
            """
            INSERT INTO filter_results (
                filter_run_id, result_id, source_detection_id, image_rel_path, predicted_label,
                keep, duplicate_group_id, kept_result_id, keep_score, similarity_to_kept,
                max_similarity, max_iou, max_containment, drop_reason, suspect_reason, area_ratio
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [(filter_run_id, item.result_id, item.source_detection_id, item.image_rel_path, item.predicted_label, 1 if item.keep else 0, item.duplicate_group_id, item.kept_result_id, item.keep_score, item.similarity_to_kept, item.max_similarity, item.max_iou, item.max_containment, item.drop_reason, item.suspect_reason, item.area_ratio) for item in decisions],
        )
        conn.commit()
    finally:
        conn.close()


def write_suspect_db(db_path: Path, *, filter_run_id: str, source_db_path: Path, semantic_run_id: str, options: dict[str, Any], boxes_by_id: dict[int, SourceBox], decisions_by_id: dict[int, FilterDecision], suspect_pairs: list[tuple[SpatialPair, float, str]]) -> None:
    conn = connect_output(db_path)
    try:
        ensure_suspect_schema(conn)
        suspect_box_ids = set()
        for pair, _, _ in suspect_pairs:
            suspect_box_ids.add(pair.a_result_id)
            suspect_box_ids.add(pair.b_result_id)
        conn.execute(
            """
            INSERT INTO suspect_runs (
                filter_run_id, created_at_utc, source_db_path, source_semantic_run_id,
                options_json, suspect_pairs, suspect_boxes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (filter_run_id, datetime.now(timezone.utc).replace(microsecond=0).isoformat(), str(source_db_path), semantic_run_id, json.dumps(options, ensure_ascii=False, sort_keys=True), len(suspect_pairs), len(suspect_box_ids)),
        )
        rows = []
        for pair, similarity, reason in suspect_pairs:
            box_a = boxes_by_id[pair.a_result_id]
            box_b = boxes_by_id[pair.b_result_id]
            decision_a = decisions_by_id.get(pair.a_result_id)
            decision_b = decisions_by_id.get(pair.b_result_id)
            rows.append((filter_run_id, box_a.image_rel_path, box_a.predicted_label, box_b.predicted_label, pair.a_result_id, pair.b_result_id, reason, pair.iou, pair.containment, similarity, box_a.area_ratio, box_b.area_ratio, 1 if decision_a is None or decision_a.keep else 0, 1 if decision_b is None or decision_b.keep else 0))
        conn.executemany(
            """
            INSERT INTO suspect_pairs (
                filter_run_id, image_rel_path, predicted_label_a, predicted_label_b,
                result_id_a, result_id_b, reason, iou, containment, similarity,
                area_ratio_a, area_ratio_b, keep_a, keep_b
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter duplicate OpenCLIP semantic boxes using spatial overlap rules.")
    parser.add_argument("--db", required=True, help="Source step2 damage_scan.sqlite3.")
    parser.add_argument("--image-root", default="", help="Image root override, usually /path/to/HinhAnh.")
    parser.add_argument("--output-dir", default="", help="Output dir. Default: source DB folder / step3_spatial_filter.")
    parser.add_argument("--semantic-run-id", default="latest", help="Semantic run id, or latest.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="DINOv2 HF model id or local model folder.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-confidence-pct", type=float, default=0.0)
    parser.add_argument("--labels", default=",".join(LABELS), help="Comma-separated labels to process. Empty = all.")
    parser.add_argument("--limit-images", type=int, default=0, help="Debug mode: process top N images by box count.")
    parser.add_argument("--image-rel-path", default="", help="Process one specific image_rel_path from the source DB.")
    parser.add_argument("--padding-ratio", type=float, default=0.05)
    parser.add_argument("--group-by-label", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spatial-only-labels", default="crack,mold,spall", help="Labels filtered only by geometry overlap. Empty = none.")
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument("--containment-threshold", type=float, default=0.70)
    parser.add_argument("--similarity-threshold", type=float, default=0.92)
    parser.add_argument("--contained-similarity-threshold", type=float, default=0.88)
    parser.add_argument("--crack-similarity-threshold", type=float, default=0.98, help="Only merge crack boxes when crops are almost identical.")
    parser.add_argument("--large-area-threshold", type=float, default=0.30)
    parser.add_argument("--large-contains-min", type=int, default=3)
    parser.add_argument("--multi-feature-container-suppression", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multi-feature-containment-threshold", type=float, default=0.85)
    parser.add_argument("--multi-feature-min-groups", type=int, default=2)
    parser.add_argument("--final-spatial-suppression", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--final-iou-threshold", type=float, default=0.30)
    parser.add_argument("--final-containment-threshold", type=float, default=0.70)
    parser.add_argument("--log-every", type=int, default=25)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    source_db = Path(args.db).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve() if str(args.image_root or "").strip() else None
    output_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir or "").strip() else source_db.parent / "step3_spatial_filter"
    filtered_db = output_dir / "filtered.sqlite3"
    suspect_db = output_dir / "suspect.sqlite3"
    filter_run_id = uuid.uuid4().hex

    source_conn = connect_readonly(source_db)
    try:
        semantic_run_id = resolve_semantic_run_id(source_conn, args.semantic_run_id)
        boxes = read_source_boxes(
            source_conn,
            semantic_run_id=semantic_run_id,
            min_confidence_pct=float(args.min_confidence_pct),
            labels=parse_labels(args.labels),
            limit_images=int(args.limit_images),
            image_rel_path=str(args.image_rel_path),
        )
    finally:
        source_conn.close()
    if not boxes:
        raise RuntimeError("No source boxes matched the current filters.")

    options = {
        "min_confidence_pct": float(args.min_confidence_pct),
        "labels": parse_labels(args.labels),
        "limit_images": int(args.limit_images),
        "image_rel_path": str(args.image_rel_path),
        "padding_ratio": float(args.padding_ratio),
        "group_by_label": bool(args.group_by_label),
        "spatial_only_labels": parse_labels(args.spatial_only_labels),
        "iou_threshold": float(args.iou_threshold),
        "containment_threshold": float(args.containment_threshold),
        "similarity_threshold": float(args.similarity_threshold),
        "contained_similarity_threshold": float(args.contained_similarity_threshold),
        "crack_similarity_threshold": float(args.crack_similarity_threshold),
        "large_area_threshold": float(args.large_area_threshold),
        "large_contains_min": int(args.large_contains_min),
        "multi_feature_container_suppression": bool(args.multi_feature_container_suppression),
        "multi_feature_containment_threshold": float(args.multi_feature_containment_threshold),
        "multi_feature_min_groups": int(args.multi_feature_min_groups),
        "final_spatial_suppression": bool(args.final_spatial_suppression),
        "final_iou_threshold": float(args.final_iou_threshold),
        "final_containment_threshold": float(args.final_containment_threshold),
    }
    spatial_only_labels = {item.lower() for item in parse_labels(args.spatial_only_labels)}
    needs_dinov2 = any(box.predicted_label.lower() not in spatial_only_labels for box in boxes)
    print(f"filter_run_id={filter_run_id} boxes={len(boxes)} semantic_run_id={semantic_run_id} model={args.model_name}", flush=True)
    embedder = DinoV2Embedder(model_name=str(args.model_name), device=str(args.device)) if needs_dinov2 else None
    decisions, suspect_pairs = process_boxes(
        boxes,
        embedder=embedder,
        image_root=image_root,
        batch_size=int(args.batch_size),
        padding_ratio=float(args.padding_ratio),
        group_by_label=bool(args.group_by_label),
        iou_threshold=float(args.iou_threshold),
        containment_threshold=float(args.containment_threshold),
        similarity_threshold=float(args.similarity_threshold),
        contained_similarity_threshold=float(args.contained_similarity_threshold),
        crack_similarity_threshold=float(args.crack_similarity_threshold),
        large_area_threshold=float(args.large_area_threshold),
        large_contains_min=int(args.large_contains_min),
        spatial_only_labels=spatial_only_labels,
        log_every=int(args.log_every),
    )
    boxes_by_id = {box.result_id: box for box in boxes}
    if bool(args.multi_feature_container_suppression):
        before_kept = sum(1 for item in decisions if item.keep)
        decisions = apply_multi_feature_container_suppression(
            decisions,
            boxes_by_id,
            containment_threshold=float(args.multi_feature_containment_threshold),
            min_feature_groups=int(args.multi_feature_min_groups),
            spatial_only_labels=spatial_only_labels,
        )
        after_kept = sum(1 for item in decisions if item.keep)
        print(f"multi_feature_container_suppression dropped={before_kept - after_kept} kept={after_kept}", flush=True)
    if bool(args.final_spatial_suppression):
        before_kept = sum(1 for item in decisions if item.keep)
        decisions = apply_final_spatial_suppression(
            decisions,
            boxes_by_id,
            group_by_label=bool(args.group_by_label),
            iou_threshold=float(args.final_iou_threshold),
            containment_threshold=float(args.final_containment_threshold),
        )
        after_kept = sum(1 for item in decisions if item.keep)
        print(f"final_spatial_suppression dropped={before_kept - after_kept} kept={after_kept}", flush=True)
    decisions_by_id = {decision.result_id: decision for decision in decisions}
    write_filtered_db(filtered_db, filter_run_id=filter_run_id, source_db_path=source_db, semantic_run_id=semantic_run_id, model_name=str(args.model_name), device=(embedder.device if embedder is not None else "not_loaded"), options=options, decisions=decisions)
    write_suspect_db(suspect_db, filter_run_id=filter_run_id, source_db_path=source_db, semantic_run_id=semantic_run_id, options=options, boxes_by_id=boxes_by_id, decisions_by_id=decisions_by_id, suspect_pairs=suspect_pairs)
    kept = sum(1 for item in decisions if item.keep)
    dropped = len(decisions) - kept
    suspect_box_count = sum(1 for item in decisions if item.suspect_reason)
    print(f"filtered_db={filtered_db}", flush=True)
    print(f"suspect_db={suspect_db}", flush=True)
    print(f"total={len(decisions)} kept={kept} dropped={dropped} suspect_boxes={suspect_box_count} suspect_pairs={len(suspect_pairs)}", flush=True)
    for label in sorted({item.predicted_label for item in decisions}):
        label_items = [item for item in decisions if item.predicted_label == label]
        label_kept = sum(1 for item in label_items if item.keep)
        print(f"label={label} total={len(label_items)} kept={label_kept} dropped={len(label_items) - label_kept}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
