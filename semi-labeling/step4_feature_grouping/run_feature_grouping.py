#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
import uuid
from pathlib import Path

import numpy as np

from clusterer import build_feature_groups, to_numpy
from cropper import crop_box
from embedder import DinoV2Embedder
from models import FeatureGroupConfig, KeptBox
from output_store import write_feature_groups
from source_store import connect_readonly, load_kept_boxes, resolve_filter_run_id


DEFAULT_MODEL_NAME = "facebook/dinov2-large"
LABELS = ("crack", "mold", "spall")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_source_db() -> Path:
    return repo_root().parent / "infer_results" / "semi-labeling" / "2_sematic" / "damage_scan.sqlite3"


def parse_labels(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(raw or "").split(",") if item.strip())


def source_db_from_filtered(filtered_db_path: Path, filter_run_id: str) -> Path:
    conn = connect_readonly(filtered_db_path)
    try:
        row = conn.execute(
            "SELECT source_db_path FROM filter_runs WHERE filter_run_id = ?",
            (filter_run_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise RuntimeError(f"filter_run_id not found in filtered DB: {filter_run_id}")
    return Path(str(row["source_db_path"])).expanduser().resolve()


def embed_boxes(
    boxes: list[KeptBox],
    *,
    embedder: DinoV2Embedder,
    image_root: Path | None,
    batch_size: int,
    padding_ratio: float,
    log_every: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    total = len(boxes)
    effective_batch_size = max(1, int(batch_size))
    for start in range(0, total, effective_batch_size):
        end = min(total, start + effective_batch_size)
        if log_every > 0 and (start == 0 or start % log_every == 0):
            print(f"embedding crops {start + 1}-{end}/{total}", flush=True)
        crops = [crop_box(box, image_root, padding_ratio=padding_ratio) for box in boxes[start:end]]
        try:
            rows.append(to_numpy(embedder.embed(crops, batch_size=effective_batch_size)))
        finally:
            for crop in crops:
                crop.close()
    return np.vstack(rows) if rows else np.empty((0, 0), dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Group kept step3 boxes by DINOv2 features and DBSCAN-style clustering.")
    parser.add_argument("--source-db", default="", help="Source step2 damage_scan.sqlite3. Default: infer_results/semi-labeling/2_sematic/damage_scan.sqlite3, or inferred from filtered DB.")
    parser.add_argument("--filtered-db", default="", help="Step3 filtered.sqlite3. Default: source DB folder / step3_spatial_filter / filtered.sqlite3.")
    parser.add_argument("--image-root", default="", help="Image root override, usually /path/to/HinhAnh.")
    parser.add_argument("--output-dir", default="", help="Output dir. Default: source DB folder / step4_feature_grouping.")
    parser.add_argument("--output-db", default="", help="Output SQLite path. Overrides --output-dir.")
    parser.add_argument("--filter-run-id", default="latest", help="Step3 filter run id, or latest.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="DINOv2 HF model id or local model folder.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--labels", default=",".join(LABELS), help="Comma-separated labels to process. Empty = all.")
    parser.add_argument("--limit", type=int, default=0, help="Debug mode: process first N kept boxes after filtering.")
    parser.add_argument("--padding-ratio", type=float, default=0.05)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--cluster-method", default="dbscan", choices=["dbscan", "agglomerative"])
    parser.add_argument("--dbscan-eps", type=float, default=0.18)
    parser.add_argument("--dbscan-min-samples", type=int, default=5)
    parser.add_argument("--agglomerative-distance-threshold", type=float, default=0.22)
    parser.add_argument("--cluster-per-label", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--label-suspect-purity-threshold", type=float, default=0.90)
    parser.add_argument("--log-every", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    source_db_arg = str(args.source_db or "").strip()
    initial_source_db = Path(source_db_arg).expanduser().resolve() if source_db_arg else default_source_db()
    filtered_db = Path(str(args.filtered_db)).expanduser().resolve() if str(args.filtered_db or "").strip() else initial_source_db.parent / "step3_spatial_filter" / "filtered.sqlite3"
    if not filtered_db.is_file():
        raise FileNotFoundError(f"Step3 filtered DB not found: {filtered_db}")
    filter_run_id = resolve_filter_run_id(filtered_db, str(args.filter_run_id))
    source_db = Path(source_db_arg).expanduser().resolve() if source_db_arg else source_db_from_filtered(filtered_db, filter_run_id)
    if not source_db.is_file():
        raise FileNotFoundError(f"Source DB not found: {source_db}")

    image_root = Path(args.image_root).expanduser().resolve() if str(args.image_root or "").strip() else None
    output_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir or "").strip() else source_db.parent / "step4_feature_grouping"
    output_db = Path(args.output_db).expanduser().resolve() if str(args.output_db or "").strip() else output_dir / "feature_groups.sqlite3"
    grouping_run_id = uuid.uuid4().hex

    config = FeatureGroupConfig(
        source_db_path=source_db,
        filtered_db_path=filtered_db,
        image_root=image_root,
        output_db_path=output_db,
        filter_run_id=filter_run_id,
        model_name=str(args.model_name),
        device=str(args.device),
        batch_size=int(args.batch_size),
        labels=parse_labels(args.labels),
        limit=int(args.limit),
        padding_ratio=float(args.padding_ratio),
        pca_dim=int(args.pca_dim),
        cluster_method=str(args.cluster_method),
        dbscan_eps=float(args.dbscan_eps),
        dbscan_min_samples=int(args.dbscan_min_samples),
        agglomerative_distance_threshold=float(args.agglomerative_distance_threshold),
        cluster_per_label=bool(args.cluster_per_label),
        label_suspect_purity_threshold=float(args.label_suspect_purity_threshold),
    )

    boxes = load_kept_boxes(
        source_db_path=source_db,
        filtered_db_path=filtered_db,
        filter_run_id=filter_run_id,
        labels=config.labels,
        limit=config.limit,
    )
    if not boxes:
        raise RuntimeError("No kept boxes matched the current filters.")

    print(
        f"grouping_run_id={grouping_run_id} boxes={len(boxes)} filter_run_id={filter_run_id} model={config.model_name}",
        flush=True,
    )
    embedder = DinoV2Embedder(model_name=config.model_name, device=config.device)
    embeddings = embed_boxes(
        boxes,
        embedder=embedder,
        image_root=image_root,
        batch_size=config.batch_size,
        padding_ratio=config.padding_ratio,
        log_every=int(args.log_every),
    )
    assignments, summaries = build_feature_groups(boxes, embeddings, config)
    write_feature_groups(
        output_db,
        grouping_run_id=grouping_run_id,
        config=config,
        device=embedder.device,
        assignments=assignments,
        summaries=summaries,
    )

    outliers = sum(1 for item in assignments if item.is_outlier)
    label_suspects = sum(1 for item in assignments if item.label_suspect)
    print(f"output_db={output_db}", flush=True)
    print(f"clusters={len(summaries)} boxes={len(assignments)} outliers={outliers} label_suspect_boxes={label_suspects}", flush=True)
    for label in sorted({item.label_scope for item in assignments}):
        label_items = [item for item in assignments if item.label_scope == label]
        label_clusters = {item.cluster_key for item in label_items}
        label_outliers = sum(1 for item in label_items if item.is_outlier)
        print(f"scope={label} boxes={len(label_items)} clusters={len(label_clusters)} outliers={label_outliers}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
