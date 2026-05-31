#!/usr/bin/env python3
"""Step 06 — Reliability Scoring.

Combines semantic + core + prototype + consistency + geometry signals into a
single reliability score per detection. Optional. Run after step04/step05.
Does not fabricate missing signals.

Inputs:  semantic/core/prototype tables in resemi.sqlite3
Outputs: reliability_scoring_runs, reliability_scores (+ synced semantic_decisions)
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from shared.runtime import bootstrap

bootstrap.ensure_on_path()

from shared.runtime.paths import default_resemi_db  # noqa: E402
from .reliability_scoring import ReliabilityConfig, persist_reliability_scoring_result, run_reliability_scoring  # noqa: E402
from shared.db.schema import connect_output, utc_now  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute resemi reliability scores from semantic, core, prototype, consistency, and geometry signals.")
    parser.add_argument("--db", default=str(default_resemi_db()), help="Resemi SQLite DB.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--core-mining-run-id", default="latest", help="Core mining run id, latest, or none.")
    parser.add_argument("--prototype-score-run-id", default="latest", help="Prototype score run id, latest, or none.")
    parser.add_argument("--accept-threshold", type=float, default=0.75)
    parser.add_argument("--suspect-threshold", type=float, default=0.50)
    parser.add_argument("--strong-margin-threshold", type=float, default=0.10)
    parser.add_argument("--prototype-reject-threshold", type=float, default=0.80)
    parser.add_argument("--require-external-evidence-for-auto", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--update-decisions", action=argparse.BooleanOptionalAction, default=True)
    return parser


def print_summary(result) -> None:
    print(f"reliability_run_id={result.reliability_run_id}")
    print(f"run_id={result.run_id}")
    print(f"core_mining_run_id={result.core_mining_run_id}")
    print(f"prototype_score_run_id={result.prototype_score_run_id}")
    print(f"scored_count={len(result.scores)}")
    print(f"auto_accept_count={result.auto_accept_count}")
    print(f"suspect_count={result.suspect_count}")
    print(f"reject_count={result.reject_count}")
    if result.scores:
        min_score = min(item.reliability_score for item in result.scores)
        max_score = max(item.reliability_score for item in result.scores)
        avg_score = sum(item.reliability_score for item in result.scores) / len(result.scores)
        print(f"score_min={min_score:.6f}")
        print(f"score_avg={avg_score:.6f}")
        print(f"score_max={max_score:.6f}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = Path(args.db).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Resemi DB not found: {db_path}")
    conn = connect_output(db_path)
    try:
        config = ReliabilityConfig(
            accept_threshold=float(args.accept_threshold),
            suspect_threshold=float(args.suspect_threshold),
            strong_margin_threshold=float(args.strong_margin_threshold),
            prototype_reject_threshold=float(args.prototype_reject_threshold),
            require_external_evidence_for_auto=bool(args.require_external_evidence_for_auto),
        )
        result = run_reliability_scoring(
            conn,
            run_id=str(args.run_id),
            config=config,
            core_mining_run_id=str(args.core_mining_run_id or "none"),
            prototype_score_run_id=str(args.prototype_score_run_id or "none"),
        )
        if not bool(args.dry_run):
            persist_reliability_scoring_result(conn, result, created_at_utc=utc_now(), update_decisions=bool(args.update_decisions))
        print_summary(result)
        print(f"dry_run={bool(args.dry_run)}")
        print(f"update_decisions={bool(args.update_decisions)}")
        print(f"db={db_path}")
        return 0
    except sqlite3.Error:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
