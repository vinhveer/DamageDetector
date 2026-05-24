#!/usr/bin/env python3
"""Apply trained classifier to needs_split clusters + export complete labels."""
from __future__ import annotations

import argparse
import csv
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


def pick_latest(folder: Path, glob: str) -> Path:
    files = sorted(folder.glob(glob), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No files match {folder}/{glob}")
    return files[0]


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


def fetch_member_ids(conn: sqlite3.Connection, cluster_run_id: str, cluster_id: int) -> list[int]:
    rows = conn.execute(
        """
        SELECT result_id, image_rel_path, predicted_label
        FROM cluster_results
        WHERE cluster_run_id = ? AND cluster_id = ?
        """,
        (cluster_run_id, int(cluster_id)),
    ).fetchall()
    return rows


def load_embeddings(conn: sqlite3.Connection, embedding_run_id: str, dim: int, result_ids: list[int]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    chunk = 900
    for offset in range(0, len(result_ids), chunk):
        slice_ids = result_ids[offset : offset + chunk]
        placeholders = ",".join("?" for _ in slice_ids)
        rows = conn.execute(
            f"""
            SELECT result_id, embedding_blob FROM detection_embeddings
            WHERE embedding_run_id = ? AND result_id IN ({placeholders})
            """,
            [embedding_run_id, *slice_ids],
        ).fetchall()
        for row in rows:
            vec = np.frombuffer(bytes(row["embedding_blob"]), dtype="<f4")
            if vec.size == dim:
                out[int(row["result_id"])] = vec.astype(np.float32, copy=False)
    return out


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def confidence_bucket(value: float) -> str:
    if value > 0.95: return ">0.95"
    if value > 0.80: return "0.80-0.95"
    if value > 0.50: return "0.50-0.80"
    return "<=0.50"


def print_distribution(counter: Counter, title: str, total: int) -> None:
    print(f"\n{title}:")
    for key in sorted(counter.keys(), key=lambda k: -counter[k]):
        n = counter[key]
        pct = (100.0 * n / total) if total else 0.0
        print(f"  {key:<14} {n:>6}  ({pct:5.1f}%)")


def quick_cross_val(model_path: Path, embedding_db: Path, cluster_db: Path, session: dict, n_folds: int = 3) -> dict | None:
    """Re-run CV on the labeled subset (approved + rejected) using the SAME model class as the bundle."""
    try:
        import joblib
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.base import clone
    except ImportError:
        return None

    bundle = joblib.load(model_path)
    model = bundle["model"]
    cluster_run_id = session["cluster_run_id"]

    # Collect labeled IDs (approved/rejected only — same as training)
    box_labels: dict[int, str] = {}
    cluster_conn = connect_ro(cluster_db)
    try:
        for cid, entry in session.get("clusters", {}).items():
            status = entry.get("status")
            user_class = entry.get("cluster_class")
            if status == "approved" and user_class:
                target = user_class
            elif status == "rejected":
                target = "reject"
            else:
                continue
            for row in fetch_member_ids(cluster_conn, cluster_run_id, int(cid)):
                box_labels[int(row["result_id"])] = target
    finally:
        cluster_conn.close()

    if not box_labels:
        return None

    embedding_conn = connect_ro(embedding_db)
    try:
        emb_run_id, dim = resolve_embedding_run(embedding_conn, "latest")
        emb_map = load_embeddings(embedding_conn, emb_run_id, dim, list(box_labels.keys()))
    finally:
        embedding_conn.close()

    aligned_ids = [rid for rid in box_labels if rid in emb_map]
    X = np.stack([emb_map[rid] for rid in aligned_ids])
    y = np.asarray([box_labels[rid] for rid in aligned_ids])

    print(f"\n[cv] running {n_folds}-fold CV on {X.shape[0]} samples with cloned model…")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(clone(model), X, y, cv=skf, scoring="accuracy", n_jobs=1)
    mean = float(scores.mean())
    std = float(scores.std())
    return {"folds": [float(s) for s in scores], "mean": mean, "std": std, "n_samples": int(X.shape[0])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply trained classifier to needs_split clusters + export complete labels.")
    parser.add_argument("--model", default=None, help="Path to classifier_*.joblib (default: latest in output dir).")
    parser.add_argument("--cluster-db", default=str(default_cluster_db()))
    parser.add_argument("--embedding-db", default=str(default_embedding_db()))
    parser.add_argument("--sessions-dir", default=str(default_sessions_dir()))
    parser.add_argument("--session", default=None, help="Session id or filename (default: most recent).")
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    parser.add_argument("--cv-folds", type=int, default=3, help="K for cross-validation (0 = skip).")
    parser.add_argument("--low-conf-threshold", type=float, default=0.50, help="Predictions below this become candidates for review.")
    return parser


def main(argv: list[str] | None = None) -> int:
    import joblib

    args = parser_parse(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model).expanduser().resolve() if args.model else pick_latest(output_dir, "classifier_*.joblib")
    sessions_dir = Path(args.sessions_dir).expanduser().resolve()
    session_path = pick_session(sessions_dir, args.session)
    print(f"[model]   {model_path.name}")
    print(f"[session] {session_path.name}")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    classes = list(bundle["classes"])
    print(f"[model]   classes={classes}  (kind={bundle.get('model_kind')})")

    session = json.loads(session_path.read_text(encoding="utf-8"))
    cluster_run_id = session["cluster_run_id"]

    # Collect needs_split member IDs grouped by cluster
    cluster_conn = connect_ro(Path(args.cluster_db).expanduser().resolve())
    try:
        needs_split: list[tuple[int, int, str, str]] = []  # (cluster_id, result_id, image_rel_path, predicted_label)
        direct_user: list[tuple[int, int, str, str, str]] = []  # (cluster_id, result_id, image_rel_path, predicted_label, final_class)
        for cid_str, entry in session.get("clusters", {}).items():
            cid = int(cid_str)
            status = entry.get("status")
            user_class = entry.get("cluster_class")
            rows = fetch_member_ids(cluster_conn, cluster_run_id, cid)
            if status == "needs_split":
                for row in rows:
                    needs_split.append((cid, int(row["result_id"]), str(row["image_rel_path"]), str(row["predicted_label"])))
            elif status == "approved" and user_class:
                for row in rows:
                    direct_user.append((cid, int(row["result_id"]), str(row["image_rel_path"]), str(row["predicted_label"]), str(user_class)))
            elif status == "rejected":
                for row in rows:
                    direct_user.append((cid, int(row["result_id"]), str(row["image_rel_path"]), str(row["predicted_label"]), "reject"))
            else:
                # unreviewed / approved-without-class — skip; will fall through below
                pass
    finally:
        cluster_conn.close()

    print(f"[counts]  direct_user={len(direct_user)}  needs_split={len(needs_split)}")

    if not needs_split:
        print("No needs_split boxes to predict.")
        return 0

    # Load embeddings for needs_split boxes
    embedding_conn = connect_ro(Path(args.embedding_db).expanduser().resolve())
    try:
        emb_run_id, emb_dim = resolve_embedding_run(embedding_conn, "latest")
        emb_map = load_embeddings(embedding_conn, emb_run_id, emb_dim, [t[1] for t in needs_split])
    finally:
        embedding_conn.close()

    aligned = [t for t in needs_split if t[1] in emb_map]
    missing = len(needs_split) - len(aligned)
    if missing:
        print(f"[warn]    {missing} needs_split boxes missing embeddings (skipped)")

    X = np.stack([emb_map[t[1]] for t in aligned])
    print(f"[predict] X.shape={X.shape}")

    probs = model.predict_proba(X)
    pred_idx = np.argmax(probs, axis=1)
    confidence = probs.max(axis=1)
    classes_arr = np.asarray(classes)
    preds = classes_arr[pred_idx]

    predictions: list[dict] = []
    pred_counter: Counter = Counter()
    conf_counter: Counter = Counter()
    per_class_conf: dict[str, list[float]] = {c: [] for c in classes}
    low_conf_ids: list[int] = []

    for (cid, rid, img, raw_label), pred, conf, prob_row in zip(aligned, preds, confidence, probs):
        pred = str(pred)
        predictions.append({
            "result_id": rid,
            "cluster_id": cid,
            "image_rel_path": img,
            "predicted_label_step2": raw_label,
            "predicted_class": pred,
            "confidence": float(conf),
            "probabilities": {c: float(p) for c, p in zip(classes, prob_row)},
        })
        pred_counter[pred] += 1
        conf_counter[confidence_bucket(float(conf))] += 1
        per_class_conf[pred].append(float(conf))
        if conf < args.low_conf_threshold:
            low_conf_ids.append(rid)

    print_distribution(pred_counter, "Predicted class on needs_split", len(predictions))
    print_distribution(conf_counter, "Confidence distribution", len(predictions))
    print("\nMean confidence per predicted class:")
    for cls in classes:
        values = per_class_conf[cls]
        if not values:
            continue
        arr = np.asarray(values)
        print(f"  {cls:<10}  n={len(values):>5}  mean={arr.mean():.3f}  median={float(np.median(arr)):.3f}  min={arr.min():.3f}")
    print(f"\nLow-confidence (<{args.low_conf_threshold:.2f}) boxes: {len(low_conf_ids)}")

    # Cross-validation
    cv_result = None
    if int(args.cv_folds) >= 2:
        cv_result = quick_cross_val(
            model_path,
            Path(args.embedding_db).expanduser().resolve(),
            Path(args.cluster_db).expanduser().resolve(),
            session,
            n_folds=int(args.cv_folds),
        )
        if cv_result:
            print(f"\n[cv] {int(args.cv_folds)}-fold accuracy: {cv_result['mean']:.4f} ± {cv_result['std']:.4f} (folds={cv_result['folds']})")

    # Save predictions JSON
    run_id = uuid.uuid4().hex
    pred_path = output_dir / f"predictions_needs_split_{run_id}.json"
    payload = {
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "model_path": str(model_path),
        "session_path": str(session_path),
        "cluster_run_id": cluster_run_id,
        "embedding_run_id": emb_run_id,
        "n_predictions": len(predictions),
        "class_distribution": dict(pred_counter),
        "confidence_buckets": dict(conf_counter),
        "low_conf_threshold": float(args.low_conf_threshold),
        "low_conf_count": len(low_conf_ids),
        "cv": cv_result,
        "predictions": predictions,
    }
    pred_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Write complete labels CSV (all 30k boxes)
    complete_path = output_dir / f"complete_labels_{run_id}.csv"
    with complete_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "result_id", "image_rel_path", "cluster_id",
            "predicted_label_step2", "final_class", "source", "confidence",
        ])
        for cid, rid, img, raw_label, final in direct_user:
            source = "user_approved" if final != "reject" else "user_rejected"
            writer.writerow([rid, img, cid, raw_label, final, source, 1.0])
        for entry in predictions:
            writer.writerow([
                entry["result_id"],
                entry["image_rel_path"],
                entry["cluster_id"],
                entry["predicted_label_step2"],
                entry["predicted_class"],
                "predicted",
                round(entry["confidence"], 4),
            ])

    total_labeled = len(direct_user) + len(predictions)
    print(f"\n[saved] {pred_path.name}")
    print(f"[saved] {complete_path.name}  ({total_labeled} total labeled boxes)")
    return 0


def parser_parse(argv):
    parser = build_parser()
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
