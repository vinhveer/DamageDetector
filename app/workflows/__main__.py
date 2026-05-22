"""
Entry point: python -m workflows run <workflow_id> --values-json <file>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent  # app/
_REPO_ROOT = _APP_DIR.parent
for _p in [str(_REPO_ROOT), str(_APP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

WORKFLOWS_DIR = Path(__file__).resolve().parent


def run_workflow(workflow_id: str, values: dict) -> None:
    # Try to import run.main from the workflow's package
    import importlib
    try:
        mod = importlib.import_module(f"workflows.{workflow_id}.run")
        mod.main(values)
        return
    except ModuleNotFoundError:
        pass

    # Fallback: legacy CLI-based runner (compiled runner.py)
    try:
        from workflows import runner  # type: ignore
        runner.run(workflow_id, values)
    except Exception as exc:
        print(f"[workflows] ERROR: cannot run {workflow_id!r}: {exc}", flush=True)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a workflow by id.")
    sub = parser.add_subparsers(dest="cmd")
    run_p = sub.add_parser("run")
    run_p.add_argument("workflow_id")
    run_p.add_argument("--values-json", required=True)
    args = parser.parse_args()

    if args.cmd == "run":
        with open(args.values_json, encoding="utf-8") as f:
            values = json.load(f)
        run_workflow(args.workflow_id, values)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
