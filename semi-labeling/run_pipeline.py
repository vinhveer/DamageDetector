#!/usr/bin/env python3
"""Resemi pipeline orchestrator — the single entry point.

Commands:
  status    — show which steps have been completed for a run_id
  list-runs — list all resemi runs in the DB
  run       — invoke a specific step (or 'all' mandatory steps in order)

Usage:
  python -m resemi.run_pipeline status --run-id myrun_v1
  python -m resemi.run_pipeline list-runs
  python -m resemi.run_pipeline run step01 --run-id myrun_v1
  python -m resemi.run_pipeline run step03 --run-id myrun_v1 --view-name tight --resume
  python -m resemi.run_pipeline run all --run-id myrun_v1     # step01→02→03

Layout:
  resemi/lib/    pure logic (no CLI)
  resemi/steps/  the 9 steps, one file each (the real CLIs)
  resemi/tools/  schema_audit, review_commit, render_bbox_overlays

Step order (mandatory = must run; optional = on demand):
  step01 Semantic Init       (mandatory)
  step02 Crop Generation     (mandatory)
  step03 DINOv2 Embedding    (mandatory)  ← gate for step04-09
  step04 Core Mining         (optional)
  step05 Prototype Bank      (optional)   ← human picks prototypes
  step06 Reliability Scoring (optional)
  step07 Decision Policy     (optional)   ← splits auto_accept vs review_queue
  step08 Classifier          (optional)
  step09 Self-Training       (optional)   ← human audits before --apply-promotions
"""
from __future__ import annotations

import argparse
import importlib
import sqlite3
from pathlib import Path

from shared.runtime import bootstrap

bootstrap.ensure_on_path()

from shared.runtime.paths import default_resemi_db  # noqa: E402


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

STEPS: list[dict] = [
    {
        "name": "step01",
        "label": "Semantic Init",
        "module": "steps.step01_semantic.main",
        "optional": False,
        "check_sql": "SELECT COUNT(*) FROM semantic_decisions WHERE run_id = ?",
        "check_label": "semantic_decisions",
        "depends_on": [],
    },
    {
        "name": "step02",
        "label": "Crop Generation",
        "module": "steps.step02_crops.main",
        "optional": False,
        "check_sql": (
            "SELECT COUNT(*) FROM crop_views "
            "WHERE run_id = ? AND status = 'ok' AND source != 'step2_sematic'"
        ),
        "check_label": "crop_views(ok, multi-view)",
        "depends_on": ["step01"],
    },
    {
        "name": "step03",
        "label": "DINOv2 Embedding",
        "module": "steps.step03_embed.main",
        "optional": False,
        "check_sql": "SELECT COUNT(*) FROM embedding_runs WHERE run_id = ?",
        "check_label": "embedding_runs",
        "depends_on": ["step02"],
    },
    {
        "name": "step04",
        "label": "Core Cluster Mining",
        "module": "steps.step04_core.main",
        "optional": True,
        "check_sql": "SELECT COUNT(*) FROM core_mining_runs WHERE run_id = ?",
        "check_label": "core_mining_runs",
        "depends_on": ["step03"],
    },
    {
        "name": "step05",
        "label": "Prototype Bank",
        "module": "steps.step05_proto.main",
        "optional": True,
        "check_sql": "SELECT COUNT(*) FROM prototype_scoring_runs WHERE run_id = ?",
        "check_label": "prototype_scoring_runs",
        "depends_on": ["step03"],
    },
    {
        "name": "step06",
        "label": "Reliability Scoring",
        "module": "steps.step06_reliability.main",
        "optional": True,
        "check_sql": "SELECT COUNT(*) FROM reliability_scoring_runs WHERE run_id = ?",
        "check_label": "reliability_scoring_runs",
        "depends_on": ["step04", "step05"],
    },
    {
        "name": "step07",
        "label": "Decision Policy",
        "module": "steps.step07_decision.main",
        "optional": True,
        "check_sql": "SELECT COUNT(*) FROM decision_policy_runs WHERE run_id = ?",
        "check_label": "decision_policy_runs",
        "depends_on": ["step06"],
    },
    {
        "name": "step08",
        "label": "Lightweight Classifier",
        "module": "steps.step08_classifier.main",
        "optional": True,
        "check_sql": "SELECT COUNT(*) FROM classifier_runs WHERE run_id = ?",
        "check_label": "classifier_runs",
        "depends_on": ["step07"],
    },
    {
        "name": "step09",
        "label": "Self-Training",
        "module": "steps.step09_self_train.main",
        "optional": True,
        "check_sql": "SELECT COUNT(*) FROM self_training_runs WHERE run_id = ?",
        "check_label": "self_training_runs",
        "depends_on": ["step08"],
    },
]

STEP_BY_NAME = {s["name"]: s for s in STEPS}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def _step_count(conn: sqlite3.Connection, step: dict, run_id: str) -> int:
    try:
        return int(conn.execute(step["check_sql"], (run_id,)).fetchone()[0])
    except sqlite3.OperationalError:
        return 0


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> int:
    import sys
    db_path = Path(args.db).expanduser().resolve()
    if not db_path.is_file():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1

    conn = _open_db(db_path)
    try:
        run_row = conn.execute(
            "SELECT run_id, created_at_utc, source_semantic_run_id, "
            "total_detections, cleaned_count, suspect_count, reject_count "
            "FROM resemi_runs WHERE run_id = ?",
            (args.run_id,),
        ).fetchone()

        if run_row is None:
            print(f"Run not found: '{args.run_id}'")
            recent = conn.execute(
                "SELECT run_id, created_at_utc FROM resemi_runs ORDER BY created_at_utc DESC LIMIT 5"
            ).fetchall()
            if recent:
                print("\nRecent runs:")
                for row in recent:
                    print(f"  {row['run_id']}  ({row['created_at_utc']})")
            return 1

        print(f"run_id:   {run_row['run_id']}")
        print(f"created:  {run_row['created_at_utc']}")
        print(f"semantic: {run_row['source_semantic_run_id']}")
        total = run_row["total_detections"]
        if total:
            print(
                f"counts:   {total} total | "
                f"{run_row['cleaned_count']} cleaned | "
                f"{run_row['suspect_count']} suspect | "
                f"{run_row['reject_count']} reject"
            )
        print()
        print("Step status:")
        for step in STEPS:
            count = _step_count(conn, step, args.run_id)
            done = count > 0
            marker = "OK " if done else ("-- " if step["optional"] else "XX ")
            opt = " [opt]" if step["optional"] else "      "
            dep = f"  <- {', '.join(step['depends_on'])}" if step["depends_on"] else ""
            count_str = f"{step['check_label']}={count}" if done else f"not run{dep}"
            print(f"  {marker} {step['name']:8s}  {step['label']:<25s}{opt}  {count_str}")
        return 0
    finally:
        conn.close()


def cmd_list_runs(args: argparse.Namespace) -> int:
    import sys
    db_path = Path(args.db).expanduser().resolve()
    if not db_path.is_file():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1

    conn = _open_db(db_path)
    try:
        rows = conn.execute(
            "SELECT run_id, created_at_utc, source_semantic_run_id, "
            "total_detections, cleaned_count, suspect_count, reject_count "
            "FROM resemi_runs ORDER BY created_at_utc DESC"
        ).fetchall()
        if not rows:
            print("No runs found.")
            return 0
        header = f"{'run_id':<32}  {'created':<22}  {'total':>7}  {'clean':>6}  {'susp':>5}  {'rej':>5}"
        print(header)
        print("-" * len(header))
        for row in rows:
            print(
                f"{row['run_id']:<32}  {row['created_at_utc']:<22}  "
                f"{row['total_detections']:>7}  {row['cleaned_count']:>6}  "
                f"{row['suspect_count']:>5}  {row['reject_count']:>5}"
            )
        return 0
    finally:
        conn.close()


def cmd_run(args: argparse.Namespace, extra_args: list[str]) -> int:
    import sys
    step_name = str(args.step).lower()

    if step_name == "all":
        mandatory = [s for s in STEPS if not s["optional"]]
        for step in mandatory:
            print(f"\n{'-' * 60}")
            print(f"  {step['name']}: {step['label']}")
            print(f"{'-' * 60}")
            rc = _invoke_step(step, args.run_id, args.db, extra_args)
            if rc != 0:
                print(f"\nStep {step['name']} failed (exit {rc}). Stopping.", file=sys.stderr)
                return rc
        print("\nAll mandatory steps done.")
        return 0

    step = STEP_BY_NAME.get(step_name)
    if step is None:
        valid = ", ".join(s["name"] for s in STEPS) + ", all"
        print(f"Unknown step '{step_name}'. Valid: {valid}", file=sys.stderr)
        return 1

    return _invoke_step(step, args.run_id, args.db, extra_args)


def _invoke_step(step: dict, run_id: str, db: str, extra_args: list[str]) -> int:
    mod = importlib.import_module(step["module"])
    argv = list(extra_args)
    if "--run-id" not in argv and run_id:
        argv = ["--run-id", run_id] + argv
    # step01 uses --output-db, everything else uses --db
    if "--db" not in argv and "--output-db" not in argv and db and step["name"] != "step01":
        argv = ["--db", db] + argv
    return mod.main(argv)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m resemi.run_pipeline",
        description="Resemi pipeline runner — run steps and check status.",
    )
    parser.add_argument("--db", default=str(default_resemi_db()),
                        help="Resemi SQLite DB path.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Show step completion for a run.")
    p_status.add_argument("--run-id", required=True)

    sub.add_parser("list-runs", help="List all runs in the DB.")

    p_run = sub.add_parser("run", help="Run a step or 'all' mandatory steps.")
    p_run.add_argument("step", help="step01..step09 or 'all'.")
    p_run.add_argument("--run-id", default="", help="Resemi run_id to pass to the step.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    known, extra = parser.parse_known_args(argv)

    if known.command == "status":
        return cmd_status(known)
    if known.command == "list-runs":
        return cmd_list_runs(known)
    if known.command == "run":
        return cmd_run(known, extra)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
