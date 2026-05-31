#!/usr/bin/env python3
"""Step 04 — Core Cluster Mining.

Clusters cached DINOv2 embeddings per class (MiniBatchKMeans) to find dense
core clusters and outliers. Optional. Run after step03.

Inputs:  crop_embeddings in resemi.sqlite3
Outputs: core_mining_runs, core_clusters, core_cluster_members, core_outliers
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from resemi.lib import bootstrap

bootstrap.ensure_on_path()

from resemi.lib.core_mining import CoreMiningConfig, persist_core_mining_result, run_core_mining  # noqa: E402
from resemi.lib.paths import default_resemi_db  # noqa: E402
from resemi.lib.schema import connect_output, utc_now  # noqa: E402


DEFAULT_MODEL_NAME = "facebook/dinov2-small"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine semantic core clusters from cached resemi DINOv2 embeddings.")
    parser.add_argument("--db", default=str(default_resemi_db()), help="Resemi SQLite DB.")
    parser.add_argument("--run-id", required=True, help="Resemi run_id to mine.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--view-name", default="tight")
    parser.add_argument("--embedding-run-id", default="", help="Optional exact embedding_run_id. Default: latest for run/model/view.")
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--min-agreement", type=float, default=0.75)
    parser.add_argument("--core-min-size", type=int, default=10)
    parser.add_argument("--rare-cluster-min-size", type=int, default=3)
    parser.add_argument("--max-clusters-per-class", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--dry-run", action="store_true", help="Compute clusters without writing DB rows.")
    return parser


def _print_summary(result) -> None:
    print(f"core_mining_run_id={result.core_mining_run_id}")
    print(f"embedding_run_id={result.embedding_run_id}")
    print(f"run_id={result.run_id}")
    print(f"total_embeddings={result.total_embeddings}")
    print(f"clustered_count={result.clustered_count}")
    print(f"core_cluster_count={len(result.clusters)}")
    print(f"rare_count={result.rare_count}")
    print(f"noise_count={result.noise_count}")
    by_label: dict[str, int] = {}
    for cluster in result.clusters:
        by_label[cluster.label] = by_label.get(cluster.label, 0) + 1
    if by_label:
        print("clusters_by_label=" + ",".join(f"{label}:{count}" for label, count in sorted(by_label.items())))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = Path(args.db).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Resemi DB not found: {db_path}")
    conn = connect_output(db_path)
    try:
        config = CoreMiningConfig(
            model_name=str(args.model_name),
            view_name=str(args.view_name),
            embedding_run_id=str(args.embedding_run_id or ""),
            min_confidence=float(args.min_confidence),
            min_agreement=float(args.min_agreement),
            core_min_size=int(args.core_min_size),
            rare_cluster_min_size=int(args.rare_cluster_min_size),
            max_clusters_per_class=int(args.max_clusters_per_class),
            random_state=int(args.random_state),
        )
        result = run_core_mining(conn, run_id=str(args.run_id), config=config)
        if not bool(args.dry_run):
            persist_core_mining_result(conn, result, created_at_utc=utc_now())
        _print_summary(result)
        print(f"dry_run={bool(args.dry_run)}")
        print(f"db={db_path}")
        return 0
    except sqlite3.Error:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
