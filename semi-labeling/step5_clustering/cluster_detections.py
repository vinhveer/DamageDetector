#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from cluster_algo import (
    compute_distances_to_centroid,
    fit_clusters,
    rank_within_clusters,
    reduce_pca,
)
from output_store import (
    ClusterAssignment,
    connect_output,
    ensure_schema,
    insert_run_metadata,
    write_assignments,
    write_summary,
)
from source_store import (
    connect_readonly,
    load_embedding_matrix,
    read_kept_boxes,
    resolve_dedup_run_id,
    resolve_embedding_run_id,
)

DEFAULT_LABELS = "crack,spall,mold"


def default_dedup_db() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "step4_class_aware_dedup" / "dedup.sqlite3"


def default_embedding_db() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "step3_embedding" / "embeddings.sqlite3"


def default_output_db() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "step5_clustering" / "clusters.sqlite3"


def parse_labels(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-class clustering of dedup-kept boxes using DINOv2 embeddings.")
    parser.add_argument("--dedup-db", default=str(default_dedup_db()), help="Step 4 dedup.sqlite3.")
    parser.add_argument("--embedding-db", default=str(default_embedding_db()), help="Step 3 embeddings.sqlite3.")
    parser.add_argument("--output-db", default=str(default_output_db()), help="Output clusters.sqlite3 path.")
    parser.add_argument("--dedup-run-id", default="latest", help="Dedup run id, or latest.")
    parser.add_argument("--embedding-run-id", default="latest", help="Embedding run id, or latest.")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Comma-separated labels to include. Empty = all.")
    parser.add_argument("--target-cluster-size", type=int, default=100, help="Target boxes per cluster (auto K = N // this).")
    parser.add_argument("--min-clusters", type=int, default=10, help="Lower bound on K.")
    parser.add_argument("--max-clusters", type=int, default=500, help="Upper bound on K.")
    parser.add_argument("--pca-dim", type=int, default=0, help="Reduce embedding dimensionality before clustering. 0 = skip PCA (cluster on full 1536D for max quality).")
    parser.add_argument("--representatives-per-cluster", type=int, default=5, help="How many top-N closest-to-centroid boxes marked as representatives.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for PCA + KMeans (reproducibility).")
    parser.add_argument("--batch-size", type=int, default=4096, help="MiniBatchKMeans batch size.")
    parser.add_argument("--max-iter", type=int, default=300, help="MiniBatchKMeans max iterations.")
    parser.add_argument("--n-init", type=int, default=5, help="MiniBatchKMeans n_init.")
    return parser


def run(args: argparse.Namespace) -> tuple[str, int, int, float]:
    dedup_db = Path(args.dedup_db).expanduser().resolve()
    embedding_db = Path(args.embedding_db).expanduser().resolve()
    output_db = Path(args.output_db).expanduser().resolve()
    if not dedup_db.is_file():
        raise FileNotFoundError(f"Dedup DB not found: {dedup_db}")
    if not embedding_db.is_file():
        raise FileNotFoundError(f"Embedding DB not found: {embedding_db}")

    labels = parse_labels(str(args.labels))

    dedup_conn = connect_readonly(dedup_db)
    try:
        dedup_run_id = resolve_dedup_run_id(dedup_conn, str(args.dedup_run_id))
        boxes = read_kept_boxes(dedup_conn, dedup_run_id=dedup_run_id, labels=labels)
    finally:
        dedup_conn.close()
    if not boxes:
        raise RuntimeError("No kept boxes matched the filter.")

    embedding_conn = connect_readonly(embedding_db)
    try:
        embedding_run_id, embedding_dim = resolve_embedding_run_id(embedding_conn, str(args.embedding_run_id))
        matrix_1536, boxes_aligned = load_embedding_matrix(
            embedding_conn,
            embedding_run_id=embedding_run_id,
            dim=embedding_dim,
            boxes=boxes,
        )
    finally:
        embedding_conn.close()
    if matrix_1536.shape[0] == 0:
        raise RuntimeError("No embeddings found for the kept boxes.")

    n = int(matrix_1536.shape[0])
    target_size = max(1, int(args.target_cluster_size))
    raw_k = max(1, n // target_size)
    k = max(int(args.min_clusters), min(int(args.max_clusters), raw_k))
    k = max(1, min(k, n))

    print(f"[cluster] N={n} target_size={target_size} -> k={k}", flush=True)

    pca_result = reduce_pca(matrix_1536, n_components=int(args.pca_dim), random_state=int(args.random_state))
    print(f"[cluster] PCA dim={pca_result.transformed.shape[1]} explained={pca_result.explained_variance_ratio:.4f}", flush=True)

    fit = fit_clusters(
        pca_result.transformed,
        k=k,
        random_state=int(args.random_state),
        batch_size=int(args.batch_size),
        max_iter=int(args.max_iter),
        n_init=int(args.n_init),
    )
    distances = compute_distances_to_centroid(pca_result.transformed, fit.labels, fit.centroids)
    ranks = rank_within_clusters(distances, fit.labels, k=fit.k)

    reps_per_cluster = max(1, int(args.representatives_per_cluster))
    assignments: list[ClusterAssignment] = []
    for idx, box in enumerate(boxes_aligned):
        rank = int(ranks[idx])
        cluster_id = int(fit.labels[idx])
        assignments.append(
            ClusterAssignment(
                result_id=box.result_id,
                image_rel_path=box.image_rel_path,
                predicted_label=box.predicted_label,
                cluster_id=cluster_id,
                distance_to_centroid=float(distances[idx]),
                rank_in_cluster=rank,
                is_representative=(0 <= rank < reps_per_cluster),
            )
        )

    cluster_run_id = uuid.uuid4().hex
    options: dict[str, Any] = {
        "dedup_db": str(dedup_db),
        "embedding_db": str(embedding_db),
        "output_db": str(output_db),
        "dedup_run_id": str(args.dedup_run_id),
        "resolved_dedup_run_id": dedup_run_id,
        "embedding_run_id": str(args.embedding_run_id),
        "resolved_embedding_run_id": embedding_run_id,
        "labels": labels,
        "target_cluster_size": int(target_size),
        "k": int(fit.k),
        "min_clusters": int(args.min_clusters),
        "max_clusters": int(args.max_clusters),
        "pca_dim": int(pca_result.transformed.shape[1]),
        "representatives_per_cluster": int(reps_per_cluster),
        "random_state": int(args.random_state),
        "batch_size": int(args.batch_size),
        "max_iter": int(args.max_iter),
        "n_init": int(args.n_init),
        "skipped_boxes_without_embeddings": int(len(boxes) - len(boxes_aligned)),
    }

    out_conn = connect_output(output_db)
    try:
        ensure_schema(out_conn)
        insert_run_metadata(
            out_conn,
            cluster_run_id=cluster_run_id,
            dedup_db_path=dedup_db,
            dedup_run_id=dedup_run_id,
            embedding_db_path=embedding_db,
            embedding_run_id=embedding_run_id,
            algorithm="minibatch_kmeans+pca",
            options=options,
            total_boxes=len(boxes_aligned),
            total_clusters=int(fit.k),
            pca_dim=int(pca_result.transformed.shape[1]),
            pca_explained_ratio=float(pca_result.explained_variance_ratio),
        )
        write_assignments(out_conn, cluster_run_id=cluster_run_id, assignments=assignments)
        write_summary(
            out_conn,
            cluster_run_id=cluster_run_id,
            assignments=assignments,
            centroids=fit.centroids,
        )
    finally:
        out_conn.close()

    return cluster_run_id, len(boxes_aligned), int(fit.k), float(pca_result.explained_variance_ratio)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cluster_run_id, n, k, explained = run(args)
    print(
        f"cluster_run_id={cluster_run_id} boxes={n} clusters={k} pca_explained={explained:.4f} db={Path(args.output_db).expanduser().resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
