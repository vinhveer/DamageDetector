#!/usr/bin/env python3
"""Standalone linear-probe evaluation over cached DINOv2 embeddings.

Answers one question: can a simple learned classifier separate crack/mold/spall
from the DINOv2-giant embeddings already stored in the DB? This is a diagnostic
only -- it reads embeddings + cleaned labels, trains a logistic-regression probe
with a train/val split, and prints accuracy + per-class report + confusion
matrix. It does NOT write the DB or touch the dataset/pipeline.

Usage:
    python -m steps.step08_classifier.probe_eval --db pipeline.sqlite3 --run-id myrun
"""
from __future__ import annotations

import argparse

import numpy as np

from shared.runtime import bootstrap

bootstrap.ensure_on_path()

from shared.db.embedding_cache import load_embeddings  # noqa: E402
from shared.db.schema import connect_output  # noqa: E402

DEFAULT_MODEL_NAME = "facebook/dinov2-giant"
DAMAGE_LABELS = ("crack", "mold", "spall")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Linear-probe diagnostic over cached DINOv2 embeddings (read-only).")
    parser.add_argument("--db", required=True, help="Resemi SQLite DB.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--view-name", default="tight")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularisation strength for LogisticRegression.")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                        help="L2-normalise embeddings before probing (cosine space). Default on.")
    parser.add_argument("--cv-folds", type=int, default=0, help="If >1, also report stratified k-fold CV accuracy.")
    return parser


def _read_labels(conn, *, run_id: str) -> dict[int, str]:
    rows = conn.execute(
        "SELECT result_id, final_label FROM cleaned_labels WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    out: dict[int, str] = {}
    for row in rows:
        label = str(row["final_label"] or "").strip().lower()
        if label in DAMAGE_LABELS:
            out[int(row["result_id"])] = label
    return out


def main(argv: list[str] | None = None) -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import cross_val_score, train_test_split

    args = build_parser().parse_args(argv)
    conn = connect_output(args.db)
    try:
        label_by_id = _read_labels(conn, run_id=str(args.run_id))
        if not label_by_id:
            raise SystemExit(f"No cleaned damage labels found for run_id={args.run_id!r}.")
        ids_wanted = sorted(label_by_id)
        matrix, ids, run = load_embeddings(
            conn,
            model_name=str(args.model_name),
            view_name=str(args.view_name),
            run_id=str(args.run_id),
            result_ids=ids_wanted,
        )
    finally:
        conn.close()

    if matrix.shape[0] == 0:
        raise SystemExit("No embeddings matched the cleaned labels (check model-name/view-name).")

    y = np.array([label_by_id[int(rid)] for rid in ids])
    X = matrix.astype(np.float32, copy=False)
    if bool(args.normalize):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms

    print(f"embedding_run_id={run['embedding_run_id']}  dim={X.shape[1]}")
    print(f"samples={X.shape[0]} (of {len(ids_wanted)} cleaned labels)")
    unique, counts = np.unique(y, return_counts=True)
    print("class_counts=" + ", ".join(f"{label}:{int(count)}" for label, count in zip(unique, counts)))
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(args.test_size), random_state=int(args.random_state), stratify=y
    )
    clf = LogisticRegression(max_iter=1000, C=float(args.C), class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = float((y_pred == y_test).mean())
    print(f"holdout_accuracy={accuracy:.4f}  (test_size={args.test_size})")
    print()
    print("classification_report (holdout):")
    print(classification_report(y_test, y_pred, digits=3))

    labels_sorted = sorted(unique.tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    print("confusion_matrix (rows=true, cols=pred):")
    header = "        " + "  ".join(f"{label:>7s}" for label in labels_sorted)
    print(header)
    for label, row in zip(labels_sorted, cm):
        print(f"{label:>7s}  " + "  ".join(f"{int(val):7d}" for val in row))

    if int(args.cv_folds) > 1:
        scores = cross_val_score(
            LogisticRegression(max_iter=1000, C=float(args.C), class_weight="balanced"),
            X, y, cv=int(args.cv_folds), scoring="accuracy",
        )
        print()
        print(f"cv{int(args.cv_folds)}_accuracy={scores.mean():.4f} +/- {scores.std():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
