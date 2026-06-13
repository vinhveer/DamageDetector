#!/usr/bin/env python3
"""Step 07 — Decision Policy.

Applies the audited decision rules and rebuilds cleaned_labels + review_queue.
Optional. Run after step06. Splits detections into auto_accept vs review_queue.

Inputs:  reliability_scores in resemi.sqlite3
Outputs: decision_policy_runs, decision_policy_audit, cleaned_labels, review_queue
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from shared.runtime import bootstrap

bootstrap.ensure_on_path()

from .decision_policy_v1 import DecisionPolicyConfig, apply_decision_policy, persist_decision_policy_result  # noqa: E402
from shared.runtime.paths import default_resemi_db  # noqa: E402
from shared.db.schema import connect_output, utc_now  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply audited resemi decision policy and rebuild cleaned/review label tables.")
    parser.add_argument("--db", default=str(default_resemi_db()), help="Resemi SQLite DB.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reliability-run-id", default="latest", help="Reliability run id, latest, or none.")
    parser.add_argument("--view-name", default="tight")
    parser.add_argument("--accept-threshold", type=float, default=0.75)
    parser.add_argument("--accept-threshold-class", action="append", default=[], metavar="LABEL=VALUE",
                        help="Per-class auto-accept reliability threshold, e.g. --accept-threshold-class crack=0.75. Repeatable; overrides --accept-threshold for that label.")
    parser.add_argument("--suspect-threshold", type=float, default=0.50)
    parser.add_argument("--suspect-threshold-class", action="append", default=[], metavar="LABEL=VALUE",
                        help="Per-class reliability floor below which a box becomes suspect, e.g. --suspect-threshold-class spall=0.3. Repeatable; overrides --suspect-threshold for that label.")
    parser.add_argument("--prototype-min-sim", type=float, default=0.70)
    parser.add_argument("--prototype-min-sim-class", action="append", default=[], metavar="LABEL=VALUE",
                        help="Per-class prototype/core similarity threshold, e.g. --prototype-min-sim-class spall=0.5. Repeatable; overrides --prototype-min-sim for that label.")
    parser.add_argument("--relabel-margin", type=float, default=0.05)
    parser.add_argument("--ambiguous-margin", type=float, default=0.03)
    parser.add_argument("--allow-low-priority-cleaned", action=argparse.BooleanOptionalAction, default=False,
                        help="Export uncertain low-priority items as cleaned. Default is off: uncertain items stay in review_queue.")
    parser.add_argument("--allow-low-priority-class", action="append", default=[], metavar="LABEL=VALUE",
                        help="Per-class low-priority-cleaned switch, e.g. --allow-low-priority-class spall=1 mold=0. Repeatable; overrides --allow-low-priority-cleaned for that label.")
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
        def _parse_by_class(items: list[str], flag: str) -> dict[str, float]:
            out: dict[str, float] = {}
            for item in items:
                label, _, value = str(item).partition("=")
                label = label.strip()
                if not label or not value.strip():
                    raise SystemExit(f"{flag} expects LABEL=VALUE, got: {item!r}")
                out[label] = float(value)
            return out

        def _parse_bool_by_class(items: list[str], flag: str) -> dict[str, bool]:
            out: dict[str, bool] = {}
            for item in items:
                label, _, value = str(item).partition("=")
                label = label.strip()
                value = value.strip().lower()
                if not label or not value:
                    raise SystemExit(f"{flag} expects LABEL=VALUE, got: {item!r}")
                if value in {"1", "true", "yes", "on"}:
                    out[label] = True
                elif value in {"0", "false", "no", "off"}:
                    out[label] = False
                else:
                    raise SystemExit(f"{flag} value must be 1/0/true/false, got: {item!r}")
            return out

        min_sim_by_class = _parse_by_class(args.prototype_min_sim_class, "--prototype-min-sim-class")
        suspect_by_class = _parse_by_class(args.suspect_threshold_class, "--suspect-threshold-class")
        accept_by_class = _parse_by_class(args.accept_threshold_class, "--accept-threshold-class")
        low_priority_by_class = _parse_bool_by_class(args.allow_low_priority_class, "--allow-low-priority-class")
        config = DecisionPolicyConfig(
            accept_threshold=float(args.accept_threshold),
            suspect_threshold=float(args.suspect_threshold),
            prototype_min_sim=float(args.prototype_min_sim),
            relabel_margin=float(args.relabel_margin),
            ambiguous_margin=float(args.ambiguous_margin),
            view_name=str(args.view_name),
            allow_low_priority_cleaned=bool(args.allow_low_priority_cleaned),
            prototype_min_sim_by_class=min_sim_by_class or None,
            suspect_threshold_by_class=suspect_by_class or None,
            accept_threshold_by_class=accept_by_class or None,
            allow_low_priority_by_class=low_priority_by_class or None,
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
