#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def _resolve_lab_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "DamageDetector").exists() and (candidate / "infer_results").exists():
            return candidate
    return current.parents[3]


def _prepare_imports() -> None:
    package_parent = Path(__file__).resolve().parents[1]
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))


LAB_ROOT = _resolve_lab_root()
_prepare_imports()

from resemi.decision_policy_v1 import DecisionPolicyConfig, apply_decision_policy, persist_decision_policy_result  # noqa: E402
from resemi.schema import connect_output, utc_now  # noqa: E402


def default_db() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "resemi" / "resemi.sqlite3"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply audited resemi decision policy and rebuild cleaned/review label tables.")
    parser.add_argument("--db", default=str(default_db()), help="Resemi SQLite DB.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reliability-run-id", default="latest", help="Reliability run id, latest, or none.")
    parser.add_argument("--view-name", default="tight")
    parser.add_argument("--accept-threshold", type=float, default=0.75)
    parser.add_argument("--suspect-threshold", type=float, default=0.50)
    parser.add_argument("--prototype-min-sim", type=float, default=0.70)
    parser.add_argument("--relabel-margin", type=float, default=0.05)
    parser.add_argument("--ambiguous-margin", type=float, default=0.03)
    parser.add_argument("--allow-low-priority-cleaned", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def print_summary(result) -> None:
    counts: dict[str, int] = {}
    rules: dict[str, int] = {}
    for decision in result.decisions:
        counts[decision.final_decision_type] = counts.get(decision.final_decision_type, 0) + 1
        rules[decision.matched_rule] = rules.get(decision.matched_rule, 0) + 1
    print(f"decision_policy_run_id={result.decision_policy_run_id}")
    print(f"run_id={result.run_id}")
    print(f"reliability_run_id={result.reliability_run_id}")
    print(f"taxonomy_version_id={result.taxonomy_version_id}")
    print(f"total_count={len(result.decisions)}")
    print(f"cleaned_count={result.cleaned_count}")
    print(f"review_count={result.review_count}")
    print(f"reject_count={result.reject_count}")
    print("decisions_by_type=" + ",".join(f"{key}:{value}" for key, value in sorted(counts.items())))
    print("rules=" + ",".join(f"{key}:{value}" for key, value in sorted(rules.items())))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = Path(args.db).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Resemi DB not found: {db_path}")
    conn = connect_output(db_path)
    try:
        config = DecisionPolicyConfig(
            accept_threshold=float(args.accept_threshold),
            suspect_threshold=float(args.suspect_threshold),
            prototype_min_sim=float(args.prototype_min_sim),
            relabel_margin=float(args.relabel_margin),
            ambiguous_margin=float(args.ambiguous_margin),
            view_name=str(args.view_name),
            allow_low_priority_cleaned=bool(args.allow_low_priority_cleaned),
        )
        result = apply_decision_policy(conn, run_id=str(args.run_id), config=config, reliability_run_id=str(args.reliability_run_id or "none"))
        if not bool(args.dry_run):
            persist_decision_policy_result(conn, result, created_at_utc=utc_now())
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
