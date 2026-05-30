#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np


def resolve_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "object_detection").exists() and (candidate / "tools").exists():
            return candidate
    return current.parents[2]


REPO_ROOT = resolve_repo_root()
LAB_ROOT = REPO_ROOT.parent
STEP_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))

from greedy_dedup import greedy_dedup
from cross_class import apply_cross_class_containment
from output_store import (
    DedupDecision,
    connect_output,
    ensure_schema,
    finalize_run_counts,
    insert_run_metadata,
    persist_pair_scores,
    write_decisions,
)
from post_filters import apply_oversized_drop, apply_prime_box
from source_store import (
    align_embeddings,
    connect_readonly,
    groupby_image,
    limit_to_first_images,
    load_embedding_map,
    read_detections,
    resolve_embedding_run_id,
    resolve_semantic_run_id,
)

DEFAULT_LABELS = "crack,spall,mold"


def default_source_db() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "step2_sematic" / "damage_scan.sqlite3"


def default_embedding_db() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "step3_embedding" / "embeddings.sqlite3"


def default_output_db() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "step4_class_aware_dedup" / "dedup.sqlite3"


def default_image_root() -> Path:
    return LAB_ROOT / "data" / "HinhAnh"


def parse_labels(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def parse_bool(raw: str | bool) -> bool:
    if isinstance(raw, bool):
        return raw
    value = str(raw or "").strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {raw}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Greedy + prime-box deduplication for Step 2 detections + Step 3 embeddings.")
    parser.add_argument("--source-db", default=str(default_source_db()), help="Source Step 2 damage_scan.sqlite3.")
    parser.add_argument("--embedding-db", default=str(default_embedding_db()), help="Step 3 embeddings.sqlite3.")
    parser.add_argument("--output-db", default=str(default_output_db()), help="Output dedup.sqlite3 path.")
    parser.add_argument("--semantic-run-id", default="latest", help="Semantic run id, or latest.")
    parser.add_argument("--embedding-run-id", default="latest", help="Embedding run id, or latest.")
    parser.add_argument("--dup-score-threshold", type=float, default=0.10, help="Greedy duplicate threshold for dup_score = IoU x cos_sim.")
    parser.add_argument("--save-pair-scores", type=parse_bool, default=True, help="Persist dedup_pair_scores for audit/labeling.")
    parser.add_argument("--image-root", default=str(default_image_root()), help="Image root directory (no longer required for the algorithm, kept for compat).")
    parser.add_argument("--limit-images", type=int, default=0, help="Debug mode: process first N images. 0 = all.")
    parser.add_argument("--min-confidence-pct", type=float, default=0.0, help="Filter source detections by semantic confidence pct.")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Comma-separated predicted labels. Empty = all.")
    parser.add_argument("--log-every", type=int, default=250, help="Print progress every N images. 0 = quiet.")
    parser.add_argument("--disable-post-filters", type=parse_bool, default=False, help="Skip oversized + prime-box post-filters.")
    parser.add_argument("--oversized-area-ratio", type=float, default=0.70, help="Force-drop keepers whose area_ratio is at least this value.")
    parser.add_argument("--container-area-ratio", type=float, default=0.30, help="Prime-box check triggers only for keepers with area_ratio at least this value.")
    parser.add_argument("--container-overlap-threshold", type=float, default=0.85, help="Child must be contained inside parent by at least this fraction to count.")
    parser.add_argument("--prime-geom-threshold", type=float, default=0.70, help="Drop parent only when union of children covers at least this fraction of parent area.")
    parser.add_argument("--prime-feat-threshold", type=float, default=0.85, help="Drop parent only when cos_sim(parent_emb, sum_child_emb) >= this threshold.")
    parser.add_argument("--cross-class-containment", type=parse_bool, default=False, help="Flag (do not drop) kept cross-class boxes strongly contained in a different-label box.")
    parser.add_argument("--cross-class-containment-threshold", type=float, default=0.85, help="Containment threshold for cross-class suspect flagging.")
    parser.add_argument("--cross-class-cos-threshold", type=float, default=0.80, help="Cosine-similarity threshold for cross-class suspect flagging.")
    return parser


def run(args: argparse.Namespace) -> tuple[str, int, int, int]:
    source_db = Path(args.source_db).expanduser().resolve()
    embedding_db = Path(args.embedding_db).expanduser().resolve()
    output_db = Path(args.output_db).expanduser().resolve()
    if not source_db.is_file():
        raise FileNotFoundError(f"Source DB not found: {source_db}")
    if not embedding_db.is_file():
        raise FileNotFoundError(f"Embedding DB not found: {embedding_db}")

    labels = parse_labels(str(args.labels))
    source_conn = connect_readonly(source_db)
    try:
        semantic_run_id = resolve_semantic_run_id(source_conn, str(args.semantic_run_id))
        detections = read_detections(
            source_conn,
            semantic_run_id=semantic_run_id,
            labels=labels,
            min_confidence_pct=float(args.min_confidence_pct),
        )
    finally:
        source_conn.close()
    detections = limit_to_first_images(detections, int(args.limit_images))
    if not detections:
        raise RuntimeError("No detections matched the current filters.")

    embedding_conn = connect_readonly(embedding_db)
    try:
        embedding_run_id, embedding_dim = resolve_embedding_run_id(embedding_conn, str(args.embedding_run_id))
        embedding_map = load_embedding_map(
            embedding_conn,
            embedding_run_id=embedding_run_id,
            dim=embedding_dim,
            result_ids=[d.result_id for d in detections],
        )
    finally:
        embedding_conn.close()

    missing_embeddings = len(detections) - len(embedding_map)
    dedup_run_id = uuid.uuid4().hex

    options: dict[str, Any] = {
        "source_db": str(source_db),
        "embedding_db": str(embedding_db),
        "output_db": str(output_db),
        "semantic_run_id": str(args.semantic_run_id),
        "resolved_semantic_run_id": semantic_run_id,
        "embedding_run_id": str(args.embedding_run_id),
        "resolved_embedding_run_id": embedding_run_id,
        "dup_score_threshold": float(args.dup_score_threshold),
        "save_pair_scores": bool(args.save_pair_scores),
        "limit_images": int(args.limit_images),
        "min_confidence_pct": float(args.min_confidence_pct),
        "labels": labels,
        "missing_embeddings": int(missing_embeddings),
        "disable_post_filters": bool(args.disable_post_filters),
        "oversized_area_ratio": float(args.oversized_area_ratio),
        "container_area_ratio": float(args.container_area_ratio),
        "container_overlap_threshold": float(args.container_overlap_threshold),
        "prime_geom_threshold": float(args.prime_geom_threshold),
        "prime_feat_threshold": float(args.prime_feat_threshold),
    }

    duplicate_classifier_json = json.dumps(
        {"mode": "greedy_containment_cos", "formula": "max(iou, containment) * max(0, cos_sim)", "threshold": float(args.dup_score_threshold)},
        sort_keys=True,
    )
    quality_classifier_json = json.dumps(
        {"mode": "linear_formula", "formula": "0.5*semantic_prob + 0.3*detector_score + 0.2*geometry_score"},
        sort_keys=True,
    )

    out_conn = connect_output(output_db)
    try:
        ensure_schema(out_conn)
        insert_run_metadata(
            out_conn,
            dedup_run_id=dedup_run_id,
            source_db_path=source_db,
            embedding_db_path=embedding_db,
            embedding_run_id=embedding_run_id,
            duplicate_classifier_json=duplicate_classifier_json,
            quality_classifier_json=quality_classifier_json,
            options=options,
            total_detections=len(detections),
        )

        decisions: list[DedupDecision] = []
        groups = groupby_image(detections)
        pair_keys_batch: list[tuple[int, int]] = []
        pair_scores_batch: list[float] = []
        pair_feature_batch: list[dict[str, float]] = []
        save_pairs = bool(args.save_pair_scores)
        for image_index, (image_rel_path, image_detections) in enumerate(groups.items(), start=1):
            image_embeddings = align_embeddings(image_detections, embedding_map, dim=embedding_dim)

            def _log_pair(kept_id: int, candidate_id: int, score: float, iou_value: float, cos_value: float) -> None:
                if not save_pairs:
                    return
                pair_keys_batch.append((kept_id, candidate_id))
                pair_scores_batch.append(float(score))
                pair_feature_batch.append({"iou": float(iou_value), "cos_sim": float(cos_value)})

            image_decisions = greedy_dedup(
                image_detections,
                image_embeddings,
                dup_threshold=float(args.dup_score_threshold),
                pair_logger=_log_pair if save_pairs else None,
            )
            decisions.extend(image_decisions)

            if save_pairs and len(pair_keys_batch) >= 5000:
                persist_pair_scores(
                    out_conn,
                    dedup_run_id=dedup_run_id,
                    pair_keys=pair_keys_batch,
                    p_dups=pair_scores_batch,
                    pair_features=pair_feature_batch,
                )
                pair_keys_batch.clear()
                pair_scores_batch.clear()
                pair_feature_batch.clear()

            if int(args.log_every) > 0 and image_index % int(args.log_every) == 0:
                print(f"[dedup] images={image_index}/{len(groups)} decisions={len(decisions)}", flush=True)

        if save_pairs and pair_keys_batch:
            persist_pair_scores(
                out_conn,
                dedup_run_id=dedup_run_id,
                pair_keys=pair_keys_batch,
                p_dups=pair_scores_batch,
                pair_features=pair_feature_batch,
            )

        if not bool(args.disable_post_filters):
            detections_by_id = {int(d.result_id): d for d in detections}
            before_kept = sum(1 for item in decisions if item.keep)
            decisions = apply_oversized_drop(
                decisions,
                detections_by_id,
                oversized_area_ratio=float(args.oversized_area_ratio),
            )
            after_oversized = sum(1 for item in decisions if item.keep)
            decisions = apply_prime_box(
                decisions,
                detections_by_id,
                embedding_map,
                container_area_ratio=float(args.container_area_ratio),
                container_overlap_threshold=float(args.container_overlap_threshold),
                prime_geom_threshold=float(args.prime_geom_threshold),
                prime_feat_threshold=float(args.prime_feat_threshold),
            )
            after_prime = sum(1 for item in decisions if item.keep)
            print(
                f"[dedup] post_filters oversized_dropped={before_kept - after_oversized} "
                f"prime_box_dropped={after_oversized - after_prime} kept={after_prime}",
                flush=True,
            )

        if bool(args.cross_class_containment):
            detections_by_id = {int(d.result_id): d for d in detections}
            decisions = apply_cross_class_containment(
                decisions,
                detections_by_id,
                embedding_map,
                containment_threshold=float(args.cross_class_containment_threshold),
                cos_sim_threshold=float(args.cross_class_cos_threshold),
            )
            suspects = sum(1 for item in decisions if item.drop_reason == "cross_class_containment_suspect")
            print(f"[dedup] cross_class_suspects={suspects}", flush=True)

        write_decisions(out_conn, dedup_run_id=dedup_run_id, decisions=decisions)
        kept, fused, dropped = finalize_run_counts(out_conn, dedup_run_id=dedup_run_id)
    finally:
        out_conn.close()
    return dedup_run_id, kept, fused, dropped


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dedup_run_id, kept, fused, dropped = run(args)
    total_outputs = kept + fused + dropped
    print(
        f"dedup_run_id={dedup_run_id} kept={kept} fused={fused} dropped={dropped} total={total_outputs} db={Path(args.output_db).expanduser().resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
