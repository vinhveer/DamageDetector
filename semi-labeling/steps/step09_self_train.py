#!/usr/bin/env python3
"""Step 09 — Self-Training.

Runs one audited self-training promotion round. Optional. Run after step08.

HUMAN GATE: by default only writes the audit. Pass --apply-promotions to
actually update semantic_decisions / cleaned_labels after reviewing the audit.

Inputs:  classifier predictions in resemi.sqlite3
Outputs: self_training_runs, self_training_promotions (+ optional label updates)
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from lib import bootstrap

bootstrap.ensure_on_path()

from lib.paths import default_resemi_db  # noqa: E402
from lib.schema import connect_output, utc_now  # noqa: E402
from lib.self_training import SelfTrainingConfig, persist_self_training_result, run_self_training  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one audited resemi self-training promotion round.")
    parser.add_argument("--db", default=str(default_resemi_db()), help="Resemi SQLite DB.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--classifier-run-id", required=True)
    parser.add_argument("--round-index", type=int, default=1)
    parser.add_argument("--classifier-confidence-threshold", type=float, default=0.90)
    parser.add_argument("--classifier-margin-threshold", type=float, default=0.10)
    parser.add_argument("--prototype-min-similarity", type=float, default=0.70)
    parser.add_argument("--core-min-similarity", type=float, default=0.70)
    parser.add_argument("--consistency-min-score", type=float, default=0.80)
    parser.add_argument("--include-low-priority", action="store_true")
    parser.add_argument("--apply-promotions", action="store_true", help="Apply promote_clean rows into semantic_decisions/cleaned_labels.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def print_summary(result) -> None:
    actions: dict[str, int] = {}
    for item in result.promotions:
        actions[item.action] = actions.get(item.action, 0) + 1
    print(f"self_training_run_id={result.self_training_run_id}")
    print(f"run_id={result.run_id}")
    print(f"classifier_run_id={result.classifier_run_id}")
    print(f"round_index={result.config.round_index}")
    print(f"candidate_count={result.candidate_count}")
    print(f"promoted_count={result.promoted_count}")
    print(f"rejected_count={result.rejected_count}")
    print(f"deferred_count={result.deferred_count}")
    print("actions=" + ",".join(f"{key}:{value}" for key, value in sorted(actions.items())))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = Path(args.db).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Resemi DB not found: {db_path}")
    conn = connect_output(db_path)
    try:
        config = SelfTrainingConfig(
            round_index=int(args.round_index),
            classifier_confidence_threshold=float(args.classifier_confidence_threshold),
            classifier_margin_threshold=float(args.classifier_margin_threshold),
            prototype_min_similarity=float(args.prototype_min_similarity),
            core_min_similarity=float(args.core_min_similarity),
            consistency_min_score=float(args.consistency_min_score),
            include_low_priority=bool(args.include_low_priority),
            apply_promotions=bool(args.apply_promotions),
        )
        result = run_self_training(conn, run_id=str(args.run_id), classifier_run_id=str(args.classifier_run_id), config=config)
        if not bool(args.dry_run):
            persist_self_training_result(conn, result, created_at_utc=utc_now())
        print_summary(result)
        print(f"apply_promotions={bool(args.apply_promotions)}")
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
