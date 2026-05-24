#!/usr/bin/env python3
"""Cluster the CV-MLP suspects (~3,105 boxes) into small homogeneous groups
each carrying a suggested action.

Workflow:
  1. Load latest cv_oof_*.json, keep only is_suspect=True records.
  2. Group by current_label (mold/spall/crack/reject).
  3. Per current_label: UMAP 30D -> HDBSCAN leaf to get tight sub-clusters.
  4. For each suspect-cluster compute the dominant CV-predicted label
     and a suggested action (change->dominant, or reject if dominant=reject).

Output → suspect_clusters.sqlite3 under step7_label_review/.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import uuid
from collections import Counter
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


def default_step7_dir() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "step7_label_review"


def default_embedding_db() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "step3_embedding" / "embeddings.sqlite3"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def pick_latest(folder: Path, pattern: str) -> Path:
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No file matches {folder}/{pattern}")
    return files[0]


def connect_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).expanduser().resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=60.0)
    conn.row_factory = sqlite3.Row
    return conn


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


def load_embeddings_for_ids(result_ids: list[int], embedding_db_path: Path) -> tuple[str, int, dict[int, np.ndarray]]:
    conn = connect_ro(embedding_db_path)
    try:
        embedding_run_id, dim = resolve_embedding_run(conn, "latest")
        blobs: dict[int, bytes] = {}
        chunk = 900
        sorted_ids = sorted(set(result_ids))
        for offset in range(0, len(sorted_ids), chunk):
            slice_ids = sorted_ids[offset : offset + chunk]
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

    out: dict[int, np.ndarray] = {}
    for rid, blob in blobs.items():
        vec = np.frombuffer(blob, dtype="<f4")
        if vec.size != dim:
            continue
        out[rid] = vec.astype(np.float32, copy=False)
    print(f"[embeddings] run_id={embedding_run_id} · loaded {len(out)} vectors (dim={dim})")
    return embedding_run_id, dim, out


def cluster_one_class(
    class_name: str,
    records: list[dict],
    embed_map: dict[int, np.ndarray],
    min_cluster_size: int,
    min_samples: int,
    umap_dim: int,
    rng: np.random.Generator,
) -> tuple[list[int], np.ndarray]:
    """Cluster the suspect records (which share current_label=class_name).

    Returns:
      result_ids   : list of result_ids in the order they were clustered
      labels       : np.ndarray of cluster labels (-1=noise)
    """
    import hdbscan
    from sklearn.metrics import pairwise_distances

    result_ids: list[int] = []
    vectors: list[np.ndarray] = []
    for rec in records:
        rid = int(rec["result_id"])
        vec = embed_map.get(rid)
        if vec is None:
            continue
        result_ids.append(rid)
        vectors.append(vec)
    if not vectors:
        return [], np.array([], dtype=int)
    X = np.stack(vectors)
    n = X.shape[0]
    if n < min_cluster_size:
        # too few suspects → all become noise/-1 (single group)
        labels = np.full(n, -1, dtype=int)
        print(f"  [{class_name}] only {n} suspects (< min_cluster) → all in noise group")
        return result_ids, labels

    if umap_dim > 0 and n > umap_dim + 5:
        import umap

        reducer = umap.UMAP(
            n_components=int(umap_dim),
            n_neighbors=min(20, max(3, n // 10)),
            metric="cosine",
            min_dist=0.0,
            random_state=int(rng.integers(0, 2**31 - 1)),
            verbose=False,
        )
        print(f"  [{class_name}] UMAP {X.shape[1]}D -> {umap_dim}D on {n} suspects ...")
        Xr = reducer.fit_transform(X).astype(np.float32, copy=False)
        D = pairwise_distances(Xr, metric="euclidean").astype(np.float32, copy=False)
    else:
        print(f"  [{class_name}] cosine distance on {n} suspects ...")
        D = pairwise_distances(X, metric="cosine").astype(np.float32, copy=False)
    np.fill_diagonal(D, 0.0)
    eff_min_cluster = max(2, min(min_cluster_size, n - 1))
    eff_min_samples = max(1, min(min_samples, eff_min_cluster))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=eff_min_cluster,
        min_samples=eff_min_samples,
        metric="precomputed",
        cluster_selection_method="leaf",
        core_dist_n_jobs=1,
    )
    labels = clusterer.fit_predict(D.astype(np.float64))
    n_clusters = len(set(int(l) for l in labels if l >= 0))
    n_noise = int((labels == -1).sum())
    print(f"  [{class_name}] -> {n_clusters} clusters · noise={n_noise} ({n_noise / n * 100:.1f}%)")
    return result_ids, labels


def suggest_action(class_name: str, cv_labels: list[str], cv_confs: list[float]) -> tuple[str, str, float]:
    """Decide the suggested action for a suspect-cluster.

    Returns (action, target_label, dominant_fraction).
      action='change' + target_label=<dominant CV-pred>  when dominant != current_label
      action='keep' + target_label=current               when dominant == current_label  (rare)
    """
    counter = Counter(cv_labels)
    if not counter:
        return "keep", class_name, 0.0
    dominant, dominant_n = counter.most_common(1)[0]
    fraction = dominant_n / max(1, len(cv_labels))
    if dominant == class_name:
        return "keep", class_name, fraction
    return "change", dominant, fraction


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS suspect_cluster_runs (
            run_id              TEXT PRIMARY KEY,
            created_at_utc      TEXT NOT NULL,
            source_oof          TEXT NOT NULL,
            embedding_run_id    TEXT NOT NULL,
            total_suspects      INTEGER NOT NULL,
            n_clusters          INTEGER NOT NULL,
            options_json        TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS suspect_cluster_results (
            run_id              TEXT NOT NULL,
            current_label       TEXT NOT NULL,
            cluster_id          INTEGER NOT NULL,
            result_id           INTEGER NOT NULL,
            is_representative   INTEGER NOT NULL,
            rank_in_cluster     INTEGER NOT NULL,
            suspicion_score     REAL NOT NULL,
            cv_predicted_label  TEXT NOT NULL,
            cv_predicted_conf   REAL NOT NULL,
            PRIMARY KEY (run_id, result_id)
        );

        CREATE INDEX IF NOT EXISTS idx_suspect_cluster
        ON suspect_cluster_results (run_id, current_label, cluster_id, rank_in_cluster);

        CREATE TABLE IF NOT EXISTS suspect_cluster_summary (
            run_id                  TEXT NOT NULL,
            current_label           TEXT NOT NULL,
            cluster_id              INTEGER NOT NULL,
            size                    INTEGER NOT NULL,
            representative_result_id INTEGER NOT NULL,
            dominant_cv_label       TEXT NOT NULL,
            dominant_cv_fraction    REAL NOT NULL,
            avg_suspicion_score     REAL NOT NULL,
            suggested_action        TEXT NOT NULL,
            suggested_target_label  TEXT NOT NULL,
            is_noise_cluster        INTEGER NOT NULL,
            PRIMARY KEY (run_id, current_label, cluster_id)
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
    parser = argparse.ArgumentParser(description="Cluster CV-MLP suspects into small homogeneous groups.")
    parser.add_argument("--cv-oof", default=None, help="Path to cv_oof_*.json. Default: latest in step7 dir.")
    parser.add_argument("--embedding-db", default=str(default_embedding_db()))
    parser.add_argument("--output-db", default=str(default_step7_dir() / "suspect_clusters.sqlite3"))
    parser.add_argument("--step7-dir", default=str(default_step7_dir()))
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--umap-dim", type=int, default=30)
    parser.add_argument("--representatives-per-cluster", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    step7_dir = Path(args.step7_dir).expanduser().resolve()
    cv_oof_path = (
        Path(args.cv_oof).expanduser().resolve()
        if args.cv_oof
        else pick_latest(step7_dir, "cv_oof_*.json")
    )
    embedding_db = Path(args.embedding_db).expanduser().resolve()
    output_db = Path(args.output_db).expanduser().resolve()
    print(f"[input] cv_oof = {cv_oof_path}")
    print(f"[input] embeddings = {embedding_db}")

    data = json.loads(cv_oof_path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    suspects = [rec for rec in records if rec.get("is_suspect")]
    print(f"[suspects] total={len(suspects)} (of {len(records)} records, acc={data.get('oof_accuracy', 0):.3f})")
    if not suspects:
        print("[error] no suspects to cluster.")
        return 1

    # Group by current_label
    by_class: dict[str, list[dict]] = {}
    for rec in suspects:
        cls = str(rec.get("current_label", "")).strip()
        if not cls:
            continue
        by_class.setdefault(cls, []).append(rec)

    suspect_ids = [int(r["result_id"]) for r in suspects]
    embedding_run_id, _dim, embed_map = load_embeddings_for_ids(suspect_ids, embedding_db)

    rng = np.random.default_rng(args.random_state)
    run_id = uuid.uuid4().hex
    print(f"[run] run_id = {run_id}")

    results_rows: list[tuple] = []
    summary_rows: list[tuple] = []
    total_clusters = 0

    for class_name in sorted(by_class.keys()):
        recs = by_class[class_name]
        result_ids, labels = cluster_one_class(
            class_name=class_name,
            records=recs,
            embed_map=embed_map,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            umap_dim=args.umap_dim,
            rng=rng,
        )
        if not result_ids:
            continue

        # rec lookup by result_id
        rec_by_id = {int(r["result_id"]): r for r in recs}

        # For each cluster, find representatives = top-K by suspicion_score desc (most confident-wrong).
        unique = sorted(set(int(l) for l in labels))
        for sub_id in unique:
            member_locals = [i for i, lab in enumerate(labels) if int(lab) == sub_id]
            if not member_locals:
                continue
            members = [(result_ids[i], rec_by_id[result_ids[i]]) for i in member_locals]
            # sort by suspicion_score desc → representatives are most-confidently-wrong
            members.sort(key=lambda kv: float(kv[1].get("suspicion_score", 0.0)), reverse=True)

            cv_labels = [str(r.get("predicted_label", "")) for _, r in members]
            cv_confs = [float(r.get("predicted_conf", 0.0)) for _, r in members]
            suspicion = [float(r.get("suspicion_score", 0.0)) for _, r in members]

            action, target_label, fraction = suggest_action(class_name, cv_labels, cv_confs)
            counter = Counter(cv_labels)
            dominant, dominant_n = counter.most_common(1)[0] if counter else ("", 0)
            avg_suspicion = float(np.mean(suspicion)) if suspicion else 0.0
            rep_id = int(members[0][0])

            for rank, (rid, rec) in enumerate(members):
                results_rows.append(
                    (
                        run_id,
                        class_name,
                        int(sub_id),
                        int(rid),
                        1 if rank < int(args.representatives_per_cluster) else 0,
                        int(rank),
                        float(rec.get("suspicion_score", 0.0)),
                        str(rec.get("predicted_label", "")),
                        float(rec.get("predicted_conf", 0.0)),
                    )
                )
            summary_rows.append(
                (
                    run_id,
                    class_name,
                    int(sub_id),
                    int(len(members)),
                    rep_id,
                    str(dominant),
                    float(fraction),
                    avg_suspicion,
                    str(action),
                    str(target_label),
                    1 if int(sub_id) == -1 else 0,
                )
            )
            if int(sub_id) != -1:
                total_clusters += 1
        print(f"  [{class_name}] wrote {sum(1 for s in unique if s != -1)} real clusters + 1 noise group")

    options = {
        "min_cluster_size": int(args.min_cluster_size),
        "min_samples": int(args.min_samples),
        "umap_dim": int(args.umap_dim),
        "representatives_per_cluster": int(args.representatives_per_cluster),
        "random_state": int(args.random_state),
    }
    out = connect_output(output_db)
    try:
        ensure_schema(out)
        out.execute(
            """
            INSERT OR REPLACE INTO suspect_cluster_runs (
                run_id, created_at_utc, source_oof, embedding_run_id,
                total_suspects, n_clusters, options_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                utc_now(),
                str(cv_oof_path),
                embedding_run_id,
                int(len(suspects)),
                int(total_clusters),
                json.dumps(options, ensure_ascii=False, sort_keys=True),
            ),
        )
        out.executemany(
            """
            INSERT OR REPLACE INTO suspect_cluster_results (
                run_id, current_label, cluster_id, result_id,
                is_representative, rank_in_cluster, suspicion_score,
                cv_predicted_label, cv_predicted_conf
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            results_rows,
        )
        out.executemany(
            """
            INSERT OR REPLACE INTO suspect_cluster_summary (
                run_id, current_label, cluster_id, size, representative_result_id,
                dominant_cv_label, dominant_cv_fraction, avg_suspicion_score,
                suggested_action, suggested_target_label, is_noise_cluster
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            summary_rows,
        )
        out.commit()
    finally:
        out.close()

    print(f"\n[saved] {output_db}")
    print(f"  · run_id   = {run_id}")
    print(f"  · clusters = {total_clusters} (excl. noise groups)")
    print(f"  · rows     = {len(results_rows)} results · {len(summary_rows)} summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
