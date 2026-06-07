#!/usr/bin/env python3
"""Reprocess existing GDINO final boxes without running GDINO again.

The tool keeps raw GDINO detections intact.  It clusters current `stage='final'`
boxes per image, fuses duplicate boxes with a WBF/box-voting style weighted
average, marks duplicate final rows as `final_dropped_migrated`, and optionally
clones the latest OpenCLIP semantic run so downstream Step01 reads only kept
boxes.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve()
_SEMI_ROOT = _THIS.parents[1]
_REPO_ROOT = _THIS.parents[2]
for _path in (str(_SEMI_ROOT), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from shared.runtime import bootstrap

bootstrap.ensure_repo_root_on_path()

from object_detection.damage_scan.geometry import (  # noqa: E402
    AdaptiveDuplicateConfig,
    adaptive_duplicate_thresholds,
    box_area_ratio,
    box_containment,
    box_iou,
)
from object_detection.damage_scan.models import Box, Detection  # noqa: E402


@dataclass(frozen=True)
class RowDet:
    detection_id: int
    image_id: int
    image_rel_path: str
    width: int
    height: int
    label: str
    prompt_key: str
    prompt_text: str
    score: float
    source: str
    box: Box
    raw: dict[str, Any]


@dataclass(frozen=True)
class ClusterDecision:
    representative_id: int
    member_ids: tuple[int, ...]
    dropped_ids: tuple[int, ...]
    fused_box: Box
    label: str
    score: float
    thresholds: dict[str, float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Migrate GDINO final boxes from existing SQLite raw/final outputs.")
    parser.add_argument("--db", required=True, help="pipeline.sqlite3 / damage_scan SQLite DB")
    parser.add_argument("--output-db", default="", help="Optional new SQLite DB to create from --db, then migrate. Source DB is not modified.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing --output-db if it already exists.")
    parser.add_argument("--source-run-id", default="latest")
    parser.add_argument("--stage", default="final")
    parser.add_argument("--dropped-stage", default="final_dropped_migrated")
    parser.add_argument("--semantic-run-id", default="latest", help="OpenCLIP run to clone. Use none to skip semantic clone.")
    parser.add_argument("--new-semantic-run-id", default="", help="Default: migrated_<old>_<stamp>")
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.0, help="0=auto per image")
    parser.add_argument("--duplicate-containment-threshold", type=float, default=0.0, help="0=auto per image")
    parser.add_argument("--duplicate-min-area-ratio", type=float, default=0.0, help="0=auto per image")
    parser.add_argument("--disable-wbf", action="store_true", help="Keep representative coordinates instead of weighted box fusion.")
    parser.add_argument("--crack-feature-aware", action=argparse.BooleanOptionalAction, default=True,
                        help="Use crack-specific hierarchical filtering: keep one continuous long crack, drop children; reject impure giant crack boxes.")
    parser.add_argument("--limit-images", type=int, default=0, help="Debug only: process first N images from the run. 0 = all.")
    parser.add_argument("--apply", action="store_true", help="Write changes. Omitted means dry-run.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_db_path = Path(args.db).expanduser().resolve()
    if not source_db_path.is_file():
        raise FileNotFoundError(f"SQLite DB not found: {source_db_path}")
    db_path = _prepare_working_db(
        source_db_path=source_db_path,
        output_db=Path(args.output_db).expanduser().resolve() if str(args.output_db or "").strip() else None,
        overwrite=bool(args.overwrite),
    )
    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    try:
        source_run_id = _resolve_source_run(conn, str(args.source_run_id))
        rows = _read_final_rows(conn, run_id=source_run_id, stage=str(args.stage), limit_images=int(args.limit_images))
        decisions = _build_decisions(
            rows,
            config=AdaptiveDuplicateConfig(
                iou_threshold=float(args.duplicate_iou_threshold),
                containment_threshold=float(args.duplicate_containment_threshold),
                min_area_ratio=float(args.duplicate_min_area_ratio),
            ),
            use_wbf=not bool(args.disable_wbf),
            crack_feature_aware=bool(args.crack_feature_aware),
        )
        dropped = {drop_id for item in decisions for drop_id in item.dropped_ids}
        kept = {item.representative_id for item in decisions if item.representative_id not in dropped}
        changed_coords = sum(1 for item in decisions if _box_changed(rows[item.representative_id].box, item.fused_box))
        old_semantic = _resolve_semantic_run(conn, str(args.semantic_run_id)) if str(args.semantic_run_id).lower() not in {"", "none"} else ""
        new_semantic = _make_new_semantic_id(old_semantic, str(args.new_semantic_run_id)) if old_semantic else ""
        print(json.dumps({
            "db": str(db_path),
            "source_db": str(source_db_path),
            "copied_to_output_db": str(db_path) != str(source_db_path),
            "source_run_id": source_run_id,
            "input_stage": str(args.stage),
            "images": len({item.image_id for item in rows.values()}),
            "final_before": len(rows),
            "final_after": len(kept),
            "dropped": len(dropped),
            "fused_coordinate_updates": changed_coords,
            "semantic_run_clone": {"from": old_semantic, "to": new_semantic} if old_semantic else None,
            "apply": bool(args.apply),
        }, ensure_ascii=False, indent=2))
        if not args.apply:
            return 0
        conn.execute("BEGIN")
        _ensure_audit_tables(conn)
        migration_id = f"gdino_box_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        _apply_detection_migration(
            conn,
            migration_id=migration_id,
            run_id=source_run_id,
            decisions=decisions,
            row_by_id=rows,
            dropped_stage=str(args.dropped_stage),
            options={
                "duplicate_iou_threshold": float(args.duplicate_iou_threshold),
                "duplicate_containment_threshold": float(args.duplicate_containment_threshold),
                "duplicate_min_area_ratio": float(args.duplicate_min_area_ratio),
                "use_wbf": not bool(args.disable_wbf),
                "crack_feature_aware": bool(args.crack_feature_aware),
            },
        )
        if old_semantic:
            _clone_semantic_run(conn, old_semantic=old_semantic, new_semantic=new_semantic, kept_ids=kept, row_by_id=rows, decisions=decisions)
        conn.commit()
        print(f"migration_id={migration_id}")
        if new_semantic:
            print(f"new_semantic_run_id={new_semantic}")
        return 0
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _resolve_source_run(conn: sqlite3.Connection, requested: str) -> str:
    if requested and requested.lower() != "latest":
        return requested
    row = conn.execute("SELECT run_id FROM runs ORDER BY created_at_utc DESC, run_id DESC LIMIT 1").fetchone()
    if row is None:
        raise RuntimeError("No GDINO run found in DB.")
    return str(row["run_id"])


def _prepare_working_db(*, source_db_path: Path, output_db: Path | None, overwrite: bool) -> Path:
    if output_db is None:
        return source_db_path
    output_db = Path(output_db).expanduser().resolve()
    output_db.parent.mkdir(parents=True, exist_ok=True)
    if output_db.exists():
        if not bool(overwrite):
            raise FileExistsError(f"Output DB already exists: {output_db}. Pass --overwrite to replace it.")
        output_db.unlink()
    src = sqlite3.connect(str(source_db_path), timeout=60)
    try:
        dst = sqlite3.connect(str(output_db), timeout=60)
        try:
            src.backup(dst)
            dst.commit()
        finally:
            dst.close()
    finally:
        src.close()
    return output_db


def _resolve_semantic_run(conn: sqlite3.Connection, requested: str) -> str:
    if requested and requested.lower() != "latest":
        row = conn.execute("SELECT semantic_run_id FROM openclip_semantic_runs WHERE semantic_run_id = ?", (requested,)).fetchone()
        if row is None:
            raise RuntimeError(f"Semantic run not found: {requested}")
        return requested
    row = conn.execute("SELECT semantic_run_id FROM openclip_semantic_runs ORDER BY created_at_utc DESC, semantic_run_id DESC LIMIT 1").fetchone()
    if row is None:
        raise RuntimeError("No OpenCLIP semantic run found. Pass --semantic-run-id none to skip clone.")
    return str(row["semantic_run_id"])


def _read_final_rows(conn: sqlite3.Connection, *, run_id: str, stage: str, limit_images: int = 0) -> dict[int, RowDet]:
    image_clause = ""
    params: list[Any] = [run_id, stage]
    if int(limit_images) > 0:
        image_rows = conn.execute(
            """
            SELECT image_id FROM detections
            WHERE run_id=? AND stage=?
            GROUP BY image_id ORDER BY image_id LIMIT ?
            """,
            (run_id, stage, int(limit_images)),
        ).fetchall()
        image_ids = [int(row["image_id"]) for row in image_rows]
        if not image_ids:
            raise RuntimeError("No image ids found for limited migration.")
        placeholders = ",".join("?" for _ in image_ids)
        image_clause = f" AND d.image_id IN ({placeholders})"
        params.extend(image_ids)
    rows = conn.execute(
        """
        SELECT d.*, i.rel_path, i.width, i.height
        FROM detections d JOIN images i ON i.image_id = d.image_id
        WHERE d.run_id = ? AND d.stage = ?
        %s
        ORDER BY d.image_id, d.detection_id
        """ % image_clause,
        params,
    ).fetchall()
    out: dict[int, RowDet] = {}
    for row in rows:
        try:
            raw = json.loads(str(row["raw_json"] or "{}"))
        except Exception:
            raw = {}
        det = RowDet(
            detection_id=int(row["detection_id"]),
            image_id=int(row["image_id"]),
            image_rel_path=str(row["rel_path"]),
            width=int(row["width"]),
            height=int(row["height"]),
            label=str(row["label"]),
            prompt_key=str(row["prompt_key"]),
            prompt_text=str(row["prompt_text"]),
            score=float(row["score"]),
            source=str(row["source"]),
            box=Box(float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
            raw=raw if isinstance(raw, dict) else {},
        )
        out[det.detection_id] = det
    if not out:
        raise RuntimeError(f"No detections found for run_id={run_id}, stage={stage}")
    return out


def _build_decisions(
    row_by_id: dict[int, RowDet],
    *,
    config: AdaptiveDuplicateConfig,
    use_wbf: bool,
    crack_feature_aware: bool,
) -> list[ClusterDecision]:
    by_image: dict[int, list[RowDet]] = {}
    for row in row_by_id.values():
        by_image.setdefault(row.image_id, []).append(row)
    decisions: list[ClusterDecision] = []
    for image_rows in by_image.values():
        pre_decisions: list[ClusterDecision] = []
        consumed_ids: set[int] = set()
        if crack_feature_aware:
            crack_rows = [item for item in image_rows if _is_crack(item)]
            if crack_rows:
                crack_decisions = _build_crack_feature_decisions(crack_rows, config=config, use_wbf=use_wbf)
                pre_decisions.extend(crack_decisions)
                consumed_ids.update(member_id for decision in crack_decisions for member_id in decision.member_ids)

        image_rows = [item for item in image_rows if item.detection_id not in consumed_ids]
        threshold_rows = _sample_for_thresholds(image_rows, limit=300)
        dets = [Detection(item.box, item.label, item.score, item.prompt_key, item.prompt_text, "final", item.source, raw=item.raw) for item in threshold_rows]
        iou_thr, contain_thr, area_ratio_thr = adaptive_duplicate_thresholds(dets, config)
        decisions.extend(pre_decisions)
        if not image_rows:
            continue
        remaining = sorted(image_rows, key=lambda item: _quality(item), reverse=True)
        while remaining:
            seed = remaining.pop(0)
            cluster = [seed]
            rest: list[RowDet] = []
            for other in remaining:
                if _is_duplicate(seed, other, iou_thr=iou_thr, contain_thr=contain_thr, area_ratio_thr=area_ratio_thr):
                    cluster.append(other)
                else:
                    rest.append(other)
            remaining = rest
            rep = max(cluster, key=lambda item: _quality(item))
            fused = _weighted_box(cluster) if use_wbf and len(cluster) > 1 else rep.box
            decisions.append(
                ClusterDecision(
                    representative_id=rep.detection_id,
                    member_ids=tuple(item.detection_id for item in cluster),
                    dropped_ids=tuple(item.detection_id for item in cluster if item.detection_id != rep.detection_id),
                    fused_box=fused,
                    label=rep.label,
                    score=max(item.score for item in cluster),
                    thresholds={"iou": iou_thr, "containment": contain_thr, "min_area_ratio": area_ratio_thr},
                )
            )
    return decisions


def _sample_for_thresholds(rows: list[RowDet], *, limit: int) -> list[RowDet]:
    if len(rows) <= int(limit):
        return list(rows)
    # Keep high-quality boxes plus an even spread so auto thresholds reflect the
    # image while avoiding O(n^2) threshold estimation on very dense images.
    half = max(1, int(limit) // 2)
    top = sorted(rows, key=lambda item: _quality(item), reverse=True)[:half]
    top_ids = {item.detection_id for item in top}
    rest = [item for item in rows if item.detection_id not in top_ids]
    step = max(1, len(rest) // max(1, int(limit) - len(top)))
    spread = rest[::step][: max(0, int(limit) - len(top))]
    return [*top, *spread]


def _build_crack_feature_decisions(
    rows: list[RowDet],
    *,
    config: AdaptiveDuplicateConfig,
    use_wbf: bool,
) -> list[ClusterDecision]:
    if not rows:
        return []
    width = max(1, int(rows[0].width))
    height = max(1, int(rows[0].height))
    structural = [item for item in rows if _is_structural_crack(item, image_width=width, image_height=height)]
    local = [item for item in rows if item.detection_id not in {s.detection_id for s in structural}]

    decisions: list[ClusterDecision] = []
    consumed: set[int] = set()

    # 1) Structural cracks: keep only if they look like one continuous feature.
    for big in sorted(structural, key=lambda item: _crack_structural_quality(item, local, width, height), reverse=True):
        if big.detection_id in consumed:
            continue
        same_scale_members = [
            other for other in structural
            if other.detection_id != big.detection_id
            and other.detection_id not in consumed
            and _same_scale_crack_duplicate(big, other)
        ]
        child_members = [
            small for small in local
            if small.detection_id not in consumed and _is_crack_child_of(small, big)
        ]
        support = _crack_support_score(big, child_members, width, height)
        if not _keep_structural_crack(big, child_members, support, width, height):
            # Giant/impure crack-like boxes are intentionally removed; local children
            # remain available so downstream can still see individual features.
            decisions.append(
                ClusterDecision(
                    representative_id=big.detection_id,
                    member_ids=(big.detection_id,),
                    dropped_ids=(big.detection_id,),
                    fused_box=big.box,
                    label=big.label,
                    score=big.score,
                    thresholds={"crack_feature_aware": 1.0, "action": "drop_impure_structural"},
                )
            )
            consumed.add(big.detection_id)
            continue

        # If a big crack is a supported continuous feature, keep it and drop its
        # child fragments. This makes each final box closer to one unique feature.
        members = [big, *same_scale_members, *child_members]
        fused = _weighted_box([big, *same_scale_members]) if use_wbf and same_scale_members else big.box
        decisions.append(
            ClusterDecision(
                representative_id=big.detection_id,
                member_ids=tuple(item.detection_id for item in members),
                dropped_ids=tuple(item.detection_id for item in members if item.detection_id != big.detection_id),
                fused_box=fused,
                label=big.label,
                score=max(item.score for item in members),
                thresholds={
                    "crack_feature_aware": 1.0,
                    "action": "keep_structural_drop_children",
                    "child_count": float(len(child_members)),
                    "support": support,
                },
            )
        )
        consumed.update(item.detection_id for item in members)

    # 2) Remaining local cracks: only dedup same-scale overlaps. Do not let small
    # crack boxes kill each other unless they are truly near-identical.
    remaining_local = [item for item in local if item.detection_id not in consumed]
    if remaining_local:
        dets = [Detection(item.box, item.label, item.score, item.prompt_key, item.prompt_text, "final", item.source, raw=item.raw) for item in _sample_for_thresholds(remaining_local, limit=300)]
        iou_thr, contain_thr, area_ratio_thr = adaptive_duplicate_thresholds(dets, config)
        remaining = sorted(remaining_local, key=lambda item: _quality(item), reverse=True)
        while remaining:
            seed = remaining.pop(0)
            cluster = [seed]
            rest: list[RowDet] = []
            for other in remaining:
                if _same_scale_crack_duplicate(seed, other, iou_thr=max(0.48, iou_thr), contain_thr=max(0.90, contain_thr), area_ratio_thr=max(0.48, area_ratio_thr)):
                    cluster.append(other)
                else:
                    rest.append(other)
            remaining = rest
            rep = max(cluster, key=lambda item: _quality(item))
            fused = _weighted_box(cluster) if use_wbf and len(cluster) > 1 else rep.box
            decisions.append(
                ClusterDecision(
                    representative_id=rep.detection_id,
                    member_ids=tuple(item.detection_id for item in cluster),
                    dropped_ids=tuple(item.detection_id for item in cluster if item.detection_id != rep.detection_id),
                    fused_box=fused,
                    label=rep.label,
                    score=max(item.score for item in cluster),
                    thresholds={"crack_feature_aware": 1.0, "action": "local_same_scale_dedup", "iou": iou_thr, "containment": contain_thr, "min_area_ratio": area_ratio_thr},
                )
            )

    return decisions


def _is_crack(item: RowDet) -> bool:
    return str(item.label or item.prompt_key or "").strip().lower() == "crack"


def _crack_shape(item: RowDet, image_width: int, image_height: int) -> tuple[float, float, float]:
    w = max(float(item.box.width), 1e-6)
    h = max(float(item.box.height), 1e-6)
    elong = max(w / h, h / w)
    long_side_ratio = max(w / max(1.0, float(image_width)), h / max(1.0, float(image_height)))
    area_ratio = float(item.box.area) / max(1.0, float(image_width) * float(image_height))
    return elong, long_side_ratio, area_ratio


def _is_structural_crack(item: RowDet, *, image_width: int, image_height: int) -> bool:
    elong, long_side_ratio, area_ratio = _crack_shape(item, image_width, image_height)
    return bool(elong >= 2.4 and long_side_ratio >= 0.10 and 0.0004 <= area_ratio <= 0.22)


def _is_crack_child_of(small: RowDet, big: RowDet) -> bool:
    if small.detection_id == big.detection_id:
        return False
    contain = box_containment(small.box, big.box)
    ratio = box_area_ratio(small.box, big.box)
    return bool(contain >= 0.72 and ratio <= 0.48)


def _same_scale_crack_duplicate(
    a: RowDet,
    b: RowDet,
    *,
    iou_thr: float = 0.50,
    contain_thr: float = 0.90,
    area_ratio_thr: float = 0.48,
) -> bool:
    ratio = box_area_ratio(a.box, b.box)
    if ratio < float(area_ratio_thr):
        return False
    return bool(box_iou(a.box, b.box) >= float(iou_thr) or box_containment(a.box, b.box) >= float(contain_thr))


def _crack_support_score(big: RowDet, children: list[RowDet], image_width: int, image_height: int) -> float:
    elong, long_side_ratio, area_ratio = _crack_shape(big, image_width, image_height)
    child_score = min(1.0, len(children) / 4.0)
    line_score = _line_consistency(children)
    shape_score = min(1.0, (elong / 8.0) * 0.55 + long_side_ratio * 0.9)
    area_penalty = max(0.0, min(1.0, (area_ratio - 0.12) / 0.10))
    return max(0.0, min(1.0, 0.30 * float(big.score) + 0.25 * child_score + 0.30 * line_score + 0.15 * shape_score - 0.20 * area_penalty))


def _keep_structural_crack(big: RowDet, children: list[RowDet], support: float, image_width: int, image_height: int) -> bool:
    elong, long_side_ratio, area_ratio = _crack_shape(big, image_width, image_height)
    if area_ratio > 0.22:
        return False
    if area_ratio > 0.14 and len(children) < 2 and float(big.score) < 0.28:
        return False
    # Keep loose: later prototype/review can still reject. We just need to avoid
    # boxes that clearly combine too many unrelated features.
    return bool(
        (support >= 0.32)
        or (float(big.score) >= 0.30 and elong >= 3.0 and long_side_ratio >= 0.12)
        or (len(children) >= 2 and elong >= 2.4)
    )


def _crack_structural_quality(item: RowDet, local: list[RowDet], image_width: int, image_height: int) -> float:
    children = [small for small in local if _is_crack_child_of(small, item)]
    return _quality(item) + 0.35 * _crack_support_score(item, children, image_width, image_height)


def _line_consistency(rows: list[RowDet]) -> float:
    if len(rows) <= 1:
        return 0.0
    centers = [((r.box.x1 + r.box.x2) / 2.0, (r.box.y1 + r.box.y2) / 2.0) for r in rows]
    xs = [p[0] for p in centers]
    ys = [p[1] for p in centers]
    range_x = max(xs) - min(xs)
    range_y = max(ys) - min(ys)
    major = max(range_x, range_y, 1e-6)
    minor = min(range_x, range_y)
    spread_score = min(1.0, major / max(1.0, sum(max(r.box.width, r.box.height) for r in rows) / len(rows)))
    alignment_score = 1.0 - min(1.0, minor / max(major, 1e-6))
    return max(0.0, min(1.0, 0.55 * alignment_score + 0.45 * spread_score))


def _is_duplicate(a: RowDet, b: RowDet, *, iou_thr: float, contain_thr: float, area_ratio_thr: float) -> bool:
    iou_value = box_iou(a.box, b.box)
    contain_value = box_containment(a.box, b.box)
    ratio_value = box_area_ratio(a.box, b.box)
    same = a.label == b.label
    if same:
        return iou_value >= iou_thr or (contain_value >= contain_thr and ratio_value >= area_ratio_thr)
    return iou_value >= max(0.42, iou_thr) or (contain_value >= contain_thr and ratio_value >= max(0.45, area_ratio_thr))


def _weighted_box(cluster: list[RowDet]) -> Box:
    total = sum(max(1e-6, _quality(item)) for item in cluster)
    return Box(
        sum(item.box.x1 * max(1e-6, _quality(item)) for item in cluster) / total,
        sum(item.box.y1 * max(1e-6, _quality(item)) for item in cluster) / total,
        sum(item.box.x2 * max(1e-6, _quality(item)) for item in cluster) / total,
        sum(item.box.y2 * max(1e-6, _quality(item)) for item in cluster) / total,
    )


def _quality(item: RowDet) -> float:
    source = 1.0 if item.source == "tile" else 0.92
    return (0.82 * max(0.0, min(1.0, item.score))) + (0.18 * source)


def _ensure_audit_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS gdino_box_migration_runs (
            migration_id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            run_id TEXT NOT NULL,
            options_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS gdino_box_migration_members (
            migration_id TEXT NOT NULL,
            representative_detection_id INTEGER NOT NULL,
            member_detection_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            fused_box_json TEXT NOT NULL,
            thresholds_json TEXT NOT NULL,
            PRIMARY KEY (migration_id, member_detection_id)
        );
        """
    )


def _apply_detection_migration(
    conn: sqlite3.Connection,
    *,
    migration_id: str,
    run_id: str,
    decisions: list[ClusterDecision],
    row_by_id: dict[int, RowDet],
    dropped_stage: str,
    options: dict[str, Any],
) -> None:
    conn.execute(
        "INSERT INTO gdino_box_migration_runs (migration_id, created_at_utc, run_id, options_json) VALUES (?, ?, ?, ?)",
        (migration_id, _utc_now(), run_id, json.dumps(options, ensure_ascii=False, sort_keys=True)),
    )
    for decision in decisions:
        rep = row_by_id[decision.representative_id]
        raw = dict(rep.raw or {})
        raw["gdino_box_migration"] = {
            "migration_id": migration_id,
            "member_ids": list(decision.member_ids),
            "dropped_ids": list(decision.dropped_ids),
            "method": "adaptive_wbf_box_voting",
            "thresholds": decision.thresholds,
        }
        conn.execute(
            """
            UPDATE detections
            SET x1=?, y1=?, x2=?, y2=?, box_w=?, box_h=?, area_px2=?, score=?, raw_json=?
            WHERE detection_id=?
            """,
            (
                decision.fused_box.x1,
                decision.fused_box.y1,
                decision.fused_box.x2,
                decision.fused_box.y2,
                decision.fused_box.width,
                decision.fused_box.height,
                decision.fused_box.area,
                decision.score,
                json.dumps(raw, ensure_ascii=False, sort_keys=True),
                decision.representative_id,
            ),
        )
        for member_id in decision.member_ids:
            action = "keep_fused_representative" if member_id == decision.representative_id else "drop_duplicate"
            conn.execute(
                """
                INSERT INTO gdino_box_migration_members (
                    migration_id, representative_detection_id, member_detection_id, action, fused_box_json, thresholds_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    migration_id,
                    decision.representative_id,
                    member_id,
                    action,
                    json.dumps(decision.fused_box.as_xyxy()),
                    json.dumps(decision.thresholds, sort_keys=True),
                ),
            )
        if decision.dropped_ids:
            placeholders = ",".join("?" for _ in decision.dropped_ids)
            conn.execute(
                f"UPDATE detections SET stage=? WHERE detection_id IN ({placeholders})",
                (dropped_stage, *decision.dropped_ids),
            )


def _clone_semantic_run(
    conn: sqlite3.Connection,
    *,
    old_semantic: str,
    new_semantic: str,
    kept_ids: set[int],
    row_by_id: dict[int, RowDet],
    decisions: list[ClusterDecision],
) -> None:
    old = conn.execute("SELECT * FROM openclip_semantic_runs WHERE semantic_run_id=?", (old_semantic,)).fetchone()
    if old is None:
        raise RuntimeError(f"Semantic run not found: {old_semantic}")
    options = _json_dict(old["options_json"])
    options["cloned_from_semantic_run_id"] = old_semantic
    options["gdino_box_migration"] = True
    conn.execute(
        """
        INSERT INTO openclip_semantic_runs (
            semantic_run_id, created_at_utc, source_db_path, source_run_id, source_stage,
            model_name, pretrained, device, prompt_config_json, options_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            new_semantic,
            _utc_now(),
            old["source_db_path"],
            old["source_run_id"],
            old["source_stage"],
            old["model_name"],
            old["pretrained"],
            old["device"],
            old["prompt_config_json"],
            json.dumps(options, ensure_ascii=False, sort_keys=True),
        ),
    )
    decision_by_rep = {item.representative_id: item for item in decisions}
    for old_row in _iter_old_semantic_results(conn, old_semantic=old_semantic, kept_ids=kept_ids):
        source_id = int(old_row["source_detection_id"])
        det = row_by_id[source_id]
        decision = decision_by_rep[source_id]
        raw = _json_dict(old_row["raw_json"])
        raw["cloned_from_result_id"] = int(old_row["result_id"])
        raw["gdino_box_migration"] = {"member_ids": list(decision.member_ids), "dropped_ids": list(decision.dropped_ids)}
        cursor = conn.execute(
            """
            INSERT INTO openclip_semantic_results (
                semantic_run_id, source_detection_id, source_run_id, image_id, image_rel_path, image_path,
                prompt_key, detector_label, detector_score, x1, y1, x2, y2, crop_path, status,
                predicted_label, predicted_probability, predicted_probability_pct, top_prompt,
                error_type, error_message, raw_json, neg_penalty_json, adjusted_scores_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_semantic,
                source_id,
                old_row["source_run_id"],
                old_row["image_id"],
                old_row["image_rel_path"],
                old_row["image_path"],
                det.prompt_key,
                det.label,
                decision.score,
                decision.fused_box.x1,
                decision.fused_box.y1,
                decision.fused_box.x2,
                decision.fused_box.y2,
                old_row["crop_path"],
                old_row["status"],
                old_row["predicted_label"],
                old_row["predicted_probability"],
                old_row["predicted_probability_pct"],
                old_row["top_prompt"],
                old_row["error_type"],
                old_row["error_message"],
                json.dumps(raw, ensure_ascii=False, sort_keys=True),
                old_row["neg_penalty_json"] if "neg_penalty_json" in old_row.keys() else None,
                old_row["adjusted_scores_json"] if "adjusted_scores_json" in old_row.keys() else None,
            ),
        )
        new_result_id = int(cursor.lastrowid)
        scores = conn.execute("SELECT label, probability, probability_pct FROM openclip_semantic_scores WHERE result_id=?", (int(old_row["result_id"]),)).fetchall()
        conn.executemany(
            "INSERT INTO openclip_semantic_scores (result_id, label, probability, probability_pct) VALUES (?, ?, ?, ?)",
            [(new_result_id, row["label"], row["probability"], row["probability_pct"]) for row in scores],
        )


def _iter_old_semantic_results(conn: sqlite3.Connection, *, old_semantic: str, kept_ids: set[int]) -> list[sqlite3.Row]:
    if not kept_ids:
        return []
    out: list[sqlite3.Row] = []
    sorted_ids = sorted(int(item) for item in kept_ids)
    chunk_size = 800
    for offset in range(0, len(sorted_ids), chunk_size):
        chunk = sorted_ids[offset : offset + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        out.extend(
            conn.execute(
                f"""
                SELECT * FROM openclip_semantic_results
                WHERE semantic_run_id=? AND source_detection_id IN ({placeholders})
                ORDER BY result_id
                """,
                (old_semantic, *chunk),
            ).fetchall()
        )
    return out


def _json_dict(raw: object) -> dict[str, Any]:
    try:
        value = json.loads(str(raw or "{}"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _make_new_semantic_id(old: str, requested: str) -> str:
    if requested.strip():
        return requested.strip()
    return f"migrated_{old[:16]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _box_changed(a: Box, b: Box) -> bool:
    return any(abs(x - y) > 1e-3 for x, y in zip(a.as_xyxy(), b.as_xyxy(), strict=True))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())