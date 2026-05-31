#!/usr/bin/env python3
"""Per-class HDBSCAN sub-clustering on labeled boxes from Step 6.

For each label (mold, spall, crack, reject), run HDBSCAN on DINOv2 embeddings
to produce sub-clusters the user can review as groups in the UI.
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def resolve_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "object_detection").exists() and (candidate / "tools").exists():
            return candidate
    return current.parents[2]


REPO_ROOT = resolve_repo_root()
LAB_ROOT = REPO_ROOT.parent


def default_step6_dir() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "step6_classifier"


def default_embedding_db() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "step3_embedding" / "embeddings.sqlite3"


def default_output_db() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "step7_label_review" / "subclusters.sqlite3"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def connect_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).expanduser().resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=60.0)
    conn.row_factory = sqlite3.Row
    return conn


def pick_latest_csv(folder: Path, pattern: str) -> Path:
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No file matches {folder}/{pattern}")
    return files[0]


def load_labels(csv_path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = str(row.get("result_id", "")).strip()
            cls = str(row.get("final_class", "")).strip()
            if not rid or not cls:
                continue
            try:
                out[int(rid)] = cls
            except ValueError:
                continue
    return out


def resolve_embedding_run(conn: sqlite3.Connection, requested: str) -> tuple[str, int]:
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
        "SELECT embedding_run_id, dim FROM embedding_runs ORDER BY created_at_utc DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("No embedding run found.")
    return str(row["embedding_run_id"]), int(row["dim"])


def load_embeddings(
    box_labels: dict[int, str],
    embedding_db_path: Path,
    requested_run: str,
) -> tuple[str, int, np.ndarray, list[int], list[str]]:
    conn = connect_ro(embedding_db_path)
    try:
        embedding_run_id, dim = resolve_embedding_run(conn, requested_run)
        result_ids = sorted(box_labels.keys())
        blobs: dict[int, bytes] = {}
        chunk = 900
        for offset in range(0, len(result_ids), chunk):
            slice_ids = result_ids[offset : offset + chunk]
            placeholders = ",".join("?" for _ in slice_ids)
            for row in conn.execute(
                f"""
                SELECT result_id, embedding_blob FROM detection_embeddings
                WHERE embedding_run_id = ? AND result_id IN ({placeholders})
                """,
                [embedding_run_id, *slice_ids],
            ).fetchall():
                blobs[int(row["result_id"])] = bytes(row["embedding_blob"])
    finally:
        conn.close()

    vectors: list[np.ndarray] = []
    aligned_ids: list[int] = []
    aligned_labels: list[str] = []
    for rid in result_ids:
        blob = blobs.get(rid)
        if blob is None:
            continue
        vec = np.frombuffer(blob, dtype="<f4")
        if vec.size != dim:
            continue
        vectors.append(vec.astype(np.float32, copy=False))
        aligned_ids.append(rid)
        aligned_labels.append(box_labels[rid])
    X = np.stack(vectors) if vectors else np.zeros((0, dim), dtype=np.float32)
    print(
        f"[embeddings] run_id={embedding_run_id} · loaded {X.shape[0]} vectors "
        f"(dim={dim}, missing={len(box_labels) - X.shape[0]})"
    )
    return embedding_run_id, dim, X, aligned_ids, aligned_labels


def cluster_one_class(
    class_name: str,
    indices: np.ndarray,
    X: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str,
    umap_dim: int,
    representatives_per_cluster: int,
    noise_representatives: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]]]:
    """Run HDBSCAN on rows of X selected by `indices`.

    Returns:
      labels   : array shape (len(indices),)  HDBSCAN labels (-1=noise)
      core_d   : array shape (len(indices),)  proxy core distance per point
      reps_by  : {sub_cluster_id: [local_index_into_indices,...]} chosen representatives
    """
    import hdbscan
    from sklearn.metrics import pairwise_distances

    sub_X = X[indices]
    n = sub_X.shape[0]
    if umap_dim > 0 and n > umap_dim + 5:
        import umap

        reducer = umap.UMAP(
            n_components=int(umap_dim),
            n_neighbors=min(30, max(5, n // 20)),
            metric="cosine",
            min_dist=0.0,
            random_state=int(rng.integers(0, 2**31 - 1)),
            verbose=False,
        )
        print(f"  [{class_name}] UMAP {sub_X.shape[1]}D -> {umap_dim}D on {n} vectors ...")
        sub_X = reducer.fit_transform(sub_X).astype(np.float32, copy=False)
        metric_label = "euclidean"
        D = pairwise_distances(sub_X, metric="euclidean").astype(np.float32, copy=False)
    else:
        metric_label = "cosine"
        print(f"  [{class_name}] computing pairwise cosine distance on {n} vectors ...")
        D = pairwise_distances(sub_X, metric="cosine").astype(np.float32, copy=False)
    np.fill_diagonal(D, 0.0)

    eff_min_cluster = max(2, min(min_cluster_size, n - 1))
    eff_min_samples = max(1, min(min_samples, eff_min_cluster))
    print(
        f"  [{class_name}] HDBSCAN min_cluster_size={eff_min_cluster} "
        f"min_samples={eff_min_samples} method={cluster_selection_method} "
        f"(precomputed {metric_label}{', umap=' + str(umap_dim) + 'D' if umap_dim > 0 else ''})"
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=eff_min_cluster,
        min_samples=eff_min_samples,
        metric="precomputed",
        cluster_selection_method=cluster_selection_method,
        core_dist_n_jobs=1,
    )
    labels = clusterer.fit_predict(D.astype(np.float64))

    # Proxy core distance: mean of k smallest pairwise distances (k = min_samples)
    k = max(2, eff_min_samples)
    partition_k = min(k, n - 1)
    if partition_k > 0:
        partitioned = np.partition(D, partition_k, axis=1)[:, :partition_k]
        core_d = partitioned.mean(axis=1).astype(np.float32)
    else:
        core_d = np.zeros(n, dtype=np.float32)

    reps_by: dict[int, list[int]] = {}
    unique_labels = sorted(set(int(l) for l in labels))
    for sub_id in unique_labels:
        member_local = np.where(labels == sub_id)[0]
        if sub_id == -1:
            n_pick = min(noise_representatives, len(member_local))
            if n_pick == 0:
                reps_by[sub_id] = []
                continue
            pick = rng.choice(member_local, size=n_pick, replace=False)
            reps_by[sub_id] = sorted(int(i) for i in pick)
        else:
            order = sorted(member_local, key=lambda i: float(core_d[i]))
            reps_by[sub_id] = order[:representatives_per_cluster]
    return labels, core_d, reps_by


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS subcluster_runs (
            subcluster_run_id        TEXT PRIMARY KEY,
            created_at_utc           TEXT NOT NULL,
            source_csv               TEXT NOT NULL,
            embedding_db_path        TEXT NOT NULL,
            embedding_run_id         TEXT NOT NULL,
            algorithm                TEXT NOT NULL,
            options_json             TEXT NOT NULL,
            total_boxes              INTEGER NOT NULL,
            n_classes                INTEGER NOT NULL,
            hdbscan_min_cluster_size INTEGER NOT NULL,
            hdbscan_min_samples      INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS subcluster_results (
            subcluster_run_id   TEXT NOT NULL,
            class_name          TEXT NOT NULL,
            sub_cluster_id      INTEGER NOT NULL,
            result_id           INTEGER NOT NULL,
            is_noise            INTEGER NOT NULL,
            is_representative   INTEGER NOT NULL,
            rank_in_cluster     INTEGER NOT NULL,
            core_distance       REAL NOT NULL,
            PRIMARY KEY (subcluster_run_id, result_id)
        );

        CREATE INDEX IF NOT EXISTS idx_subcluster_class
        ON subcluster_results (subcluster_run_id, class_name, sub_cluster_id, rank_in_cluster);

        CREATE TABLE IF NOT EXISTS subcluster_summary (
            subcluster_run_id        TEXT NOT NULL,
            class_name               TEXT NOT NULL,
            sub_cluster_id           INTEGER NOT NULL,
            size                     INTEGER NOT NULL,
            representative_result_id INTEGER NOT NULL,
            avg_intra_distance       REAL NOT NULL,
            is_noise_cluster         INTEGER NOT NULL,
            PRIMARY KEY (subcluster_run_id, class_name, sub_cluster_id)
        );
        """
    )
    conn.commit()


def connect_output(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Per-class HDBSCAN sub-clustering of labeled boxes.")
    parser.add_argument("--complete-labels-csv", default=None, help="Default: latest complete_labels_*.csv in step6 dir.")
    parser.add_argument("--step6-dir", default=str(default_step6_dir()))
    parser.add_argument("--embedding-db", default=str(default_embedding_db()))
    parser.add_argument("--embedding-run-id", default="latest")
    parser.add_argument("--output-db", default=str(default_output_db()))
    parser.add_argument("--min-cluster-size", type=int, default=15)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument(
        "--cluster-selection-method",
        choices=["eom", "leaf"],
        default="leaf",
        help="HDBSCAN cluster selection. 'leaf' gives finer-grained groups (recommended for review).",
    )
    parser.add_argument(
        "--umap-dim",
        type=int,
        default=50,
        help="UMAP reduction dim before HDBSCAN. 0 = disable (HDBSCAN on raw 1536D = ~90%% noise). Default 50 is standard.",
    )
    parser.add_argument("--representatives-per-cluster", type=int, default=5)
    parser.add_argument("--noise-representatives", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    step6_dir = Path(args.step6_dir).expanduser().resolve()
    csv_path = (
        Path(args.complete_labels_csv).expanduser().resolve()
        if args.complete_labels_csv
        else pick_latest_csv(step6_dir, "complete_labels_*.csv")
    )
    embedding_db = Path(args.embedding_db).expanduser().resolve()
    output_db = Path(args.output_db).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"complete_labels CSV not found: {csv_path}")
    if not embedding_db.is_file():
        raise FileNotFoundError(f"Embeddings DB not found: {embedding_db}")

    print(f"[input] csv = {csv_path}")
    box_labels = load_labels(csv_path)
    dist = Counter(box_labels.values())
    print(f"[labels] {len(box_labels)} boxes · distribution: {dict(dist)}")

    embedding_run_id, dim, X, aligned_ids, aligned_labels = load_embeddings(
        box_labels, embedding_db, args.embedding_run_id
    )
    if X.shape[0] == 0:
        print("[error] no embeddings loaded; aborting.")
        return 1

    rng = np.random.default_rng(args.random_state)
    aligned_labels_arr = np.asarray(aligned_labels)

    subcluster_run_id = uuid.uuid4().hex
    print(f"[run] subcluster_run_id = {subcluster_run_id}")

    results_rows: list[tuple] = []
    summary_rows: list[tuple] = []
    n_classes = 0
    classes_in_order = sorted(set(aligned_labels))
    for class_name in classes_in_order:
        idx_local = np.where(aligned_labels_arr == class_name)[0]
        if idx_local.size == 0:
            continue
        n_classes += 1
        labels, core_d, reps_by = cluster_one_class(
            class_name=class_name,
            indices=idx_local,
            X=X,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_method=args.cluster_selection_method,
            umap_dim=args.umap_dim,
            representatives_per_cluster=args.representatives_per_cluster,
            noise_representatives=args.noise_representatives,
            rng=rng,
        )

        rep_set_by_sub: dict[int, set[int]] = {
            sub_id: set(local_indices) for sub_id, local_indices in reps_by.items()
        }
        rank_lookup: dict[int, int] = {}
        for sub_id, ordered in reps_by.items():
            for rank, local_i in enumerate(ordered):
                rank_lookup[local_i] = rank

        per_sub_members: dict[int, list[int]] = defaultdict(list)
        for offset, sub_id in enumerate(labels):
            local_i = int(offset)
            global_i = int(idx_local[local_i])
            result_id = int(aligned_ids[global_i])
            is_noise = 1 if int(sub_id) == -1 else 0
            is_rep = 1 if local_i in rep_set_by_sub.get(int(sub_id), set()) else 0
            rank_in_cluster = int(rank_lookup.get(local_i, 9999))
            results_rows.append(
                (
                    subcluster_run_id,
                    class_name,
                    int(sub_id),
                    result_id,
                    is_noise,
                    is_rep,
                    rank_in_cluster,
                    float(core_d[local_i]),
                )
            )
            per_sub_members[int(sub_id)].append(local_i)

        # Per-sub-cluster summary
        n_sub_real = sum(1 for sid in per_sub_members if sid != -1)
        noise_n = len(per_sub_members.get(-1, []))
        noise_pct = (noise_n / max(1, int(labels.size))) * 100.0
        print(
            f"  [{class_name}] sub-clusters={n_sub_real} · noise={noise_n} "
            f"({noise_pct:.1f}%) · total={int(labels.size)}"
        )

        for sub_id, member_locals in per_sub_members.items():
            size = len(member_locals)
            ordered_reps = reps_by.get(sub_id, [])
            rep_local = ordered_reps[0] if ordered_reps else member_locals[0]
            rep_result_id = int(aligned_ids[int(idx_local[rep_local])])
            # avg pairwise distance approximated by mean core_d of members
            if size > 0:
                avg_intra = float(np.mean([core_d[i] for i in member_locals]))
            else:
                avg_intra = 0.0
            summary_rows.append(
                (
                    subcluster_run_id,
                    class_name,
                    int(sub_id),
                    int(size),
                    rep_result_id,
                    avg_intra,
                    1 if int(sub_id) == -1 else 0,
                )
            )

    # Write everything
    options = {
        "min_cluster_size": int(args.min_cluster_size),
        "min_samples": int(args.min_samples),
        "cluster_selection_method": str(args.cluster_selection_method),
        "umap_dim": int(args.umap_dim),
        "representatives_per_cluster": int(args.representatives_per_cluster),
        "noise_representatives": int(args.noise_representatives),
        "metric": "cosine_precomputed",
        "random_state": int(args.random_state),
    }
    out = connect_output(output_db)
    try:
        ensure_schema(out)
        out.execute(
            """
            INSERT OR REPLACE INTO subcluster_runs (
                subcluster_run_id, created_at_utc, source_csv,
                embedding_db_path, embedding_run_id,
                algorithm, options_json, total_boxes, n_classes,
                hdbscan_min_cluster_size, hdbscan_min_samples
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                subcluster_run_id,
                utc_now(),
                str(csv_path),
                str(embedding_db),
                embedding_run_id,
                "hdbscan",
                json.dumps(options, ensure_ascii=False, sort_keys=True),
                int(len(aligned_ids)),
                int(n_classes),
                int(args.min_cluster_size),
                int(args.min_samples),
            ),
        )
        out.executemany(
            """
            INSERT OR REPLACE INTO subcluster_results (
                subcluster_run_id, class_name, sub_cluster_id, result_id,
                is_noise, is_representative, rank_in_cluster, core_distance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            results_rows,
        )
        out.executemany(
            """
            INSERT OR REPLACE INTO subcluster_summary (
                subcluster_run_id, class_name, sub_cluster_id, size,
                representative_result_id, avg_intra_distance, is_noise_cluster
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            summary_rows,
        )
        out.commit()
    finally:
        out.close()

    print(f"\n[saved] {output_db}")
    print(f"  · run_id     = {subcluster_run_id}")
    print(f"  · rows write = {len(results_rows)} / summary = {len(summary_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
