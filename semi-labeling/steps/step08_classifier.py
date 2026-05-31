#!/usr/bin/env python3
"""Step 08 — Lightweight Classifier.

Trains a small classifier (LogisticRegression / LinearSVM / MLP) on audited
labels + cached embeddings, with OOF predictions. Optional. Run after step07.

Inputs:  cleaned_labels + crop_embeddings in resemi.sqlite3
Outputs: classifier_runs, classifier_predictions, classifier_oof_predictions, ...
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from lib import bootstrap

bootstrap.ensure_on_path()

from lib.lightweight_classifier import ClassifierConfig, persist_classifier_result, train_lightweight_classifier  # noqa: E402
from lib.paths import default_resemi_db  # noqa: E402
from lib.schema import connect_output, utc_now  # noqa: E402


DEFAULT_MODEL_NAME = "facebook/dinov2-small"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small classifier on audited resemi labels and cached embeddings.")
    parser.add_argument("--db", default=str(default_resemi_db()), help="Resemi SQLite DB.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--view-name", default="tight")
    parser.add_argument("--embedding-run-id", default="")
    parser.add_argument("--classifier-run-id", default="")
    parser.add_argument("--model-type", default="logistic_regression", choices=["logistic_regression", "linear_svm", "mlp"])
    parser.add_argument("--min-train-reliability", type=float, default=0.75)
    parser.add_argument("--include-low-priority", action="store_true")
    parser.add_argument("--prototype-version-id", default="")
    parser.add_argument("--core-mining-run-id", default="")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def print_summary(result) -> None:
    disagreements = sum(1 for item in result.summaries if item.disagrees_with_policy)
    print(f"classifier_run_id={result.classifier_run_id}")
    print(f"run_id={result.run_id}")
    print(f"embedding_run_id={result.embedding_run_id}")
    print(f"model_type={result.model_type}")
    print(f"train_count={len(result.training_items)}")
    print(f"prediction_count={len(result.summaries)}")
    print("labels=" + ",".join(result.labels))
    print(f"cv_status={result.evaluation.get('cv_status')}")
    print(f"effective_cv_folds={result.evaluation.get('effective_cv_folds')}")
    print(f"policy_disagreement_count={disagreements}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = Path(args.db).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Resemi DB not found: {db_path}")
    conn = connect_output(db_path)
    try:
        config = ClassifierConfig(
            run_id=str(args.run_id),
            model_name=str(args.model_name),
            view_name=str(args.view_name),
            embedding_run_id=str(args.embedding_run_id or ""),
            model_type=str(args.model_type),
            classifier_run_id=str(args.classifier_run_id or ""),
            min_train_reliability=float(args.min_train_reliability),
            include_low_priority=bool(args.include_low_priority),
            prototype_version_id=str(args.prototype_version_id or ""),
            core_mining_run_id=str(args.core_mining_run_id or ""),
            cv_folds=int(args.cv_folds),
            random_state=int(args.random_state),
        )
        result = train_lightweight_classifier(conn, config=config)
        if not bool(args.dry_run):
            persist_classifier_result(conn, result, created_at_utc=utc_now())
        print_summary(result)
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
