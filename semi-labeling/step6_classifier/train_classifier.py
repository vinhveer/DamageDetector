#!/usr/bin/env python3
"""Train damage classifier (DINOv2 embedding -> mold/spall/crack/reject) from Step 5 session labels."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
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


def default_cluster_db() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "step5_clustering" / "clusters.sqlite3"


def default_embedding_db() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "step3_embedding" / "embeddings.sqlite3"


def default_sessions_dir() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "step5_clustering" / "sessions"


def default_output_dir() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "step6_classifier"


def connect_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).expanduser().resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=60.0)
    conn.row_factory = sqlite3.Row
    return conn


def pick_session(sessions_dir: Path, requested: str | None) -> Path:
    if requested:
        path = Path(requested).expanduser()
        if not path.is_absolute():
            path = sessions_dir / requested
            if not path.exists() and not requested.endswith(".json"):
                path = sessions_dir / f"{requested}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Session file not found: {requested}")
        return path
    files = [p for p in sessions_dir.glob("*.json") if not p.name.endswith(".bak")]
    if not files:
        raise FileNotFoundError(f"No sessions found in {sessions_dir}")
    return max(files, key=lambda p: p.stat().st_mtime)


def collect_box_labels(session: dict, cluster_db_path: Path) -> dict[int, str]:
    """Return {result_id: class} based on cluster status."""
    cluster_run_id = session["cluster_run_id"]
    cluster_entries = session.get("clusters", {})

    targets: dict[int, str] = {}
    skipped_clusters = Counter()
    conn = connect_ro(cluster_db_path)
    try:
        for cluster_id_str, entry in cluster_entries.items():
            status = str(entry.get("status", "")).strip()
            user_class = entry.get("cluster_class")
            if status == "approved" and user_class:
                target = str(user_class).strip()
            elif status == "rejected":
                target = "reject"
            else:
                skipped_clusters[status or "unreviewed"] += 1
                continue
            rows = conn.execute(
                """
                SELECT result_id FROM cluster_results
                WHERE cluster_run_id = ? AND cluster_id = ?
                """,
                (cluster_run_id, int(cluster_id_str)),
            ).fetchall()
            for row in rows:
                targets[int(row["result_id"])] = target
    finally:
        conn.close()
    print(f"[labels] {len(targets)} boxes labeled, skipped clusters: {dict(skipped_clusters)}")
    return targets


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


def load_embeddings(box_labels: dict[int, str], embedding_db_path: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    conn = connect_ro(embedding_db_path)
    try:
        embedding_run_id, dim = resolve_embedding_run(conn, "latest")
        result_ids = sorted(box_labels.keys())
        rows: list[tuple[int, bytes]] = []
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
                rows.append((int(row["result_id"]), bytes(row["embedding_blob"])))
    finally:
        conn.close()

    matrix: list[np.ndarray] = []
    targets: list[str] = []
    aligned_ids: list[int] = []
    for rid, blob in rows:
        vec = np.frombuffer(blob, dtype="<f4")
        if vec.size != dim:
            continue
        matrix.append(vec.astype(np.float32, copy=False))
        targets.append(box_labels[rid])
        aligned_ids.append(rid)
    X = np.stack(matrix) if matrix else np.zeros((0, dim), dtype=np.float32)
    y = np.asarray(targets)
    print(f"[embeddings] loaded {X.shape[0]} vectors (dim={dim}, missing={len(box_labels) - X.shape[0]})")
    return X, y, aligned_ids


def train_and_eval(X: np.ndarray, y: np.ndarray, *, random_state: int = 42) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    classes = sorted(set(y))
    print(f"[split] classes={classes} · class distribution={dict(Counter(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    print(f"[split] train={X_train.shape[0]} · test={X_test.shape[0]}")

    results = {}

    print("[lr] fitting LogisticRegression…")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_report = classification_report(y_test, lr_pred, output_dict=True, zero_division=0)
    lr_cm = confusion_matrix(y_test, lr_pred, labels=classes).tolist()
    results["logreg"] = {
        "model": lr,
        "test_accuracy": float(lr_report["accuracy"]),
        "report": lr_report,
        "confusion_matrix": lr_cm,
    }
    print(f"[lr] test_accuracy={lr_report['accuracy']:.4f}")

    print("[mlp] fitting MLPClassifier(256, 128)…")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=300,
        random_state=random_state,
        n_iter_no_change=20,
        tol=1e-4,
    )
    mlp.fit(X_train, y_train)
    mlp_pred = mlp.predict(X_test)
    mlp_report = classification_report(y_test, mlp_pred, output_dict=True, zero_division=0)
    mlp_cm = confusion_matrix(y_test, mlp_pred, labels=classes).tolist()
    results["mlp"] = {
        "model": mlp,
        "test_accuracy": float(mlp_report["accuracy"]),
        "report": mlp_report,
        "confusion_matrix": mlp_cm,
    }
    print(f"[mlp] test_accuracy={mlp_report['accuracy']:.4f}")

    results["classes"] = classes
    return results


def save_outputs(
    output_dir: Path,
    results: dict,
    *,
    session_path: Path,
    embedding_run_id: str,
    cluster_run_id: str,
    n_total: int,
) -> Path:
    import joblib

    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    lr = results["logreg"]
    mlp = results["mlp"]
    classes = results["classes"]
    best_key = "mlp" if mlp["test_accuracy"] >= lr["test_accuracy"] else "logreg"
    best = results[best_key]

    model_path = output_dir / f"classifier_{run_id}.joblib"
    joblib.dump(
        {
            "model": best["model"],
            "model_kind": best_key,
            "classes": classes,
            "session_path": str(session_path),
            "embedding_run_id": embedding_run_id,
            "cluster_run_id": cluster_run_id,
            "created_at_utc": timestamp,
        },
        model_path,
    )

    metrics_path = output_dir / f"metrics_{run_id}.json"
    metrics = {
        "run_id": run_id,
        "created_at_utc": timestamp,
        "session_path": str(session_path),
        "cluster_run_id": cluster_run_id,
        "embedding_run_id": embedding_run_id,
        "n_total_labeled": int(n_total),
        "classes": classes,
        "best_model": best_key,
        "logreg": {k: v for k, v in lr.items() if k != "model"},
        "mlp": {k: v for k, v in mlp.items() if k != "model"},
        "model_path": str(model_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return model_path, metrics_path, run_id, best_key


def print_summary(results: dict) -> None:
    classes = results["classes"]
    print("\n" + "=" * 60)
    print(" Model comparison")
    print("=" * 60)
    for key in ["logreg", "mlp"]:
        info = results[key]
        report = info["report"]
        print(f"\n  {key.upper()} (accuracy = {info['test_accuracy']:.4f})")
        print(f"    {'class':<10} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>8}")
        for cls in classes:
            row = report.get(cls, {})
            print(
                f"    {cls:<10} "
                f"{row.get('precision', 0):>10.3f} "
                f"{row.get('recall', 0):>10.3f} "
                f"{row.get('f1-score', 0):>10.3f} "
                f"{int(row.get('support', 0)):>8d}"
            )
        macro = report.get("macro avg", {})
        print(f"    {'macro avg':<10} {macro.get('precision', 0):>10.3f} {macro.get('recall', 0):>10.3f} {macro.get('f1-score', 0):>10.3f}")
        print(f"    Confusion matrix ({' x '.join(classes)}):")
        for row_label, row in zip(classes, info["confusion_matrix"]):
            row_str = " ".join(f"{v:>5d}" for v in row)
            print(f"      {row_label:<10} {row_str}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train damage classifier from Step 5 session labels.")
    parser.add_argument("--cluster-db", default=str(default_cluster_db()))
    parser.add_argument("--embedding-db", default=str(default_embedding_db()))
    parser.add_argument("--sessions-dir", default=str(default_sessions_dir()))
    parser.add_argument("--session", default=None, help="Session id or filename. If empty, picks the most recent JSON.")
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    sessions_dir = Path(args.sessions_dir).expanduser().resolve()
    session_path = pick_session(sessions_dir, args.session)
    print(f"[session] {session_path}")
    session = json.loads(session_path.read_text(encoding="utf-8"))

    box_labels = collect_box_labels(session, Path(args.cluster_db).expanduser().resolve())
    if not box_labels:
        raise RuntimeError("No labeled boxes found in session.")

    X, y, _ = load_embeddings(box_labels, Path(args.embedding_db).expanduser().resolve())
    if X.shape[0] == 0:
        raise RuntimeError("No embeddings loaded for labeled boxes.")

    results = train_and_eval(X, y, random_state=int(args.random_state))
    print_summary(results)

    embedding_conn = connect_ro(Path(args.embedding_db).expanduser().resolve())
    try:
        embedding_run_id, _ = resolve_embedding_run(embedding_conn, "latest")
    finally:
        embedding_conn.close()

    output_dir = Path(args.output_dir).expanduser().resolve()
    model_path, metrics_path, run_id, best_key = save_outputs(
        output_dir,
        results,
        session_path=session_path,
        embedding_run_id=embedding_run_id,
        cluster_run_id=session["cluster_run_id"],
        n_total=int(X.shape[0]),
    )
    print(f"\n[saved] run_id={run_id} best={best_key}")
    print(f"[saved] model={model_path}")
    print(f"[saved] metrics={metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
