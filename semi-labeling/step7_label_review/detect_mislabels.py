#!/usr/bin/env python3
"""Cross-validated MLP disagreement → ranked suspect list for mislabel review."""
from __future__ import annotations

import argparse
import csv
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


def default_embedding_db() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "step3_embedding" / "embeddings.sqlite3"


def default_step6_dir() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "step6_classifier"


def default_output_dir() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "step7_label_review"


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


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def load_complete_labels(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = str(row.get("final_class", "")).strip()
            if not label:
                continue
            rows.append({
                "result_id": int(row["result_id"]),
                "image_rel_path": str(row.get("image_rel_path", "")),
                "cluster_id": int(row.get("cluster_id") or -1),
                "current_label": label,
                "source": str(row.get("source", "")),
                "source_conf": float(row.get("confidence") or 1.0),
            })
    return rows


def load_embeddings(
    conn: sqlite3.Connection, embedding_run_id: str, dim: int, result_ids: list[int]
) -> dict[int, np.ndarray]:
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


def run_cv(X: np.ndarray, y: np.ndarray, k: int, random_state: int) -> tuple[np.ndarray, list[str]]:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neural_network import MLPClassifier

    classes = sorted(set(y))
    n = X.shape[0]
    oof = np.zeros((n, len(classes)), dtype=np.float32)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    print(f"[cv] running {k}-fold StratifiedKFold on N={n} (classes={classes})")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"  · fold {fold_idx}/{k} train={len(train_idx)} val={len(val_idx)}")
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=300,
            random_state=random_state,
            n_iter_no_change=20,
            tol=1e-4,
        )
        clf.fit(X[train_idx], y[train_idx])
        # Align class order to global `classes`
        fold_classes = list(clf.classes_)
        proba = clf.predict_proba(X[val_idx])
        # Map columns to global order
        col_map = {c: i for i, c in enumerate(fold_classes)}
        aligned = np.zeros((len(val_idx), len(classes)), dtype=np.float32)
        for gi, c in enumerate(classes):
            if c in col_map:
                aligned[:, gi] = proba[:, col_map[c]]
        oof[val_idx] = aligned

    return oof, classes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CV-MLP mislabel detection.")
    parser.add_argument("--complete-labels-csv", default=None, help="Path to complete_labels_*.csv (default: latest in step6 dir)")
    parser.add_argument("--embedding-db", default=str(default_embedding_db()))
    parser.add_argument("--step6-dir", default=str(default_step6_dir()))
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    step6_dir = Path(args.step6_dir).expanduser().resolve()
    if args.complete_labels_csv:
        csv_path = Path(args.complete_labels_csv).expanduser().resolve()
    else:
        csv_path = pick_latest(step6_dir, "complete_labels_*.csv")
    print(f"[input] complete_labels = {csv_path.name}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_complete_labels(csv_path)
    if not rows:
        raise RuntimeError("No labels loaded from CSV.")
    label_dist = Counter(r["current_label"] for r in rows)
    print(f"[labels] N={len(rows)}  distribution={dict(label_dist)}")

    embedding_db = Path(args.embedding_db).expanduser().resolve()
    conn = connect_ro(embedding_db)
    try:
        emb_run_id, dim = resolve_embedding_run(conn, "latest")
        emb_map = load_embeddings(conn, emb_run_id, dim, [r["result_id"] for r in rows])
    finally:
        conn.close()

    aligned = [r for r in rows if r["result_id"] in emb_map]
    missing = len(rows) - len(aligned)
    if missing:
        print(f"[warn] {missing} rows missing embeddings (skipped)")

    X = np.stack([emb_map[r["result_id"]] for r in aligned])
    y = np.asarray([r["current_label"] for r in aligned])
    print(f"[embeddings] X.shape={X.shape}")

    oof_proba, classes = run_cv(X, y, int(args.k_folds), int(args.random_state))
    pred_idx = np.argmax(oof_proba, axis=1)
    pred_labels = np.asarray(classes)[pred_idx]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    full_records: list[dict] = []
    suspects: list[dict] = []
    suspect_per_class: Counter = Counter()
    confusion: dict[str, Counter] = {c: Counter() for c in classes}
    correct = 0

    for i, row in enumerate(aligned):
        cur = row["current_label"]
        pred = str(pred_labels[i])
        cur_conf = float(oof_proba[i, class_to_idx.get(cur, 0)]) if cur in class_to_idx else 0.0
        pred_conf = float(oof_proba[i, pred_idx[i]])
        suspicion = pred_conf - cur_conf
        is_suspect = pred != cur
        confusion[cur][pred] += 1
        if not is_suspect:
            correct += 1
        record = {
            "result_id": row["result_id"],
            "image_rel_path": row["image_rel_path"],
            "cluster_id": row["cluster_id"],
            "current_label": cur,
            "predicted_label": pred,
            "current_conf": round(cur_conf, 4),
            "predicted_conf": round(pred_conf, 4),
            "suspicion_score": round(suspicion, 4),
            "is_suspect": is_suspect,
            "source": row["source"],
            "probabilities": {c: round(float(oof_proba[i, class_to_idx[c]]), 4) for c in classes},
        }
        full_records.append(record)
        if is_suspect:
            suspects.append(record)
            suspect_per_class[cur] += 1

    suspects.sort(key=lambda r: r["suspicion_score"], reverse=True)
    acc = correct / len(aligned) if aligned else 0.0

    print(f"\n[cv] oof accuracy = {acc:.4f}  ({correct}/{len(aligned)})")
    print(f"[cv] suspects total = {len(suspects)} ({100.0 * len(suspects) / max(1, len(aligned)):.1f}%)")
    print("[cv] suspects per current_label:")
    for c in classes:
        n_in = label_dist.get(c, 0)
        n_susp = suspect_per_class.get(c, 0)
        pct = (100.0 * n_susp / n_in) if n_in else 0.0
        print(f"  {c:<10} {n_susp:>5} / {n_in:<5}  ({pct:5.1f}%)")

    print("\n[cv] confusion (rows = current → cols = predicted):")
    header = f"  {'':<10}" + "".join(f"{c:>9}" for c in classes)
    print(header)
    for c in classes:
        row_str = "".join(f"{confusion[c].get(c2, 0):>9d}" for c2 in classes)
        print(f"  {c:<10}{row_str}")

    run_id = uuid.uuid4().hex
    timestamp = utc_now()

    oof_path = output_dir / f"cv_oof_{run_id}.json"
    suspects_path = output_dir / f"suspects_{run_id}.json"
    metrics_path = output_dir / f"cv_metrics_{run_id}.json"

    base_payload = {
        "run_id": run_id,
        "created_at_utc": timestamp,
        "source_csv": str(csv_path),
        "embedding_run_id": emb_run_id,
        "k_folds": int(args.k_folds),
        "classes": classes,
        "n_total": len(aligned),
        "n_suspects": len(suspects),
        "oof_accuracy": round(acc, 4),
    }

    oof_payload = {**base_payload, "records": full_records}
    oof_path.write_text(json.dumps(oof_payload, indent=2), encoding="utf-8")

    suspects_payload = {**base_payload, "suspects": suspects}
    suspects_path.write_text(json.dumps(suspects_payload, indent=2), encoding="utf-8")

    confusion_serialized = {c: {c2: int(confusion[c].get(c2, 0)) for c2 in classes} for c in classes}
    metrics_payload = {
        **base_payload,
        "label_distribution": dict(label_dist),
        "suspect_per_class": dict(suspect_per_class),
        "confusion_current_to_predicted": confusion_serialized,
        "oof_file": str(oof_path),
        "suspects_file": str(suspects_path),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"\n[saved] {oof_path.name}")
    print(f"[saved] {suspects_path.name}")
    print(f"[saved] {metrics_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
