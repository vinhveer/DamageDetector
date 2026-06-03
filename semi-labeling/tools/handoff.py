#!/usr/bin/env python3
"""Apply JSON handoff requests produced by the semi-labeling app.

The client flow keeps human decisions as explicit JSON artifacts first, then the
app/tool reads that JSON and applies it. This makes review/prototype handoff easy
to audit and rerun.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from shared.runtime import bootstrap

bootstrap.ensure_repo_root_on_path()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply prototype/review JSON handoff request.")
    parser.add_argument("--request-json", required=True, help="Path to handoff JSON file.")
    parser.add_argument("--action", default="auto", choices=["auto", "prototype", "review"], help="Override request type.")
    parser.add_argument("--chain", action="store_true", help="For prototype: run step05 + seed + step06 + step07.")
    parser.add_argument("--seed", action="store_true", help="For prototype: run seed after step05.")
    parser.add_argument("--policy", action="store_true", help="For prototype: run step06 + step07 after seed/step05.")
    return parser


def _read_request(path: str | Path) -> dict[str, Any]:
    request_path = Path(path).expanduser().resolve()
    if not request_path.is_file():
        raise FileNotFoundError(f"handoff request not found: {request_path}")
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("handoff request must be a JSON object")
    payload["request_path"] = str(request_path)
    return payload


def _run_module(module: str, argv: list[str]) -> int:
    print("\n" + "=" * 88, flush=True)
    print(f"$ python -m {module} {' '.join(argv)}", flush=True)
    print("=" * 88, flush=True)
    mod = importlib.import_module(module)
    return int(mod.main(argv))


def _request_type(payload: dict[str, Any], override: str) -> str:
    if override != "auto":
        return override
    kind = str(payload.get("type") or payload.get("kind") or "").strip().lower()
    if kind in {"prototype", "prototype_request"}:
        return "prototype"
    if kind in {"review", "review_request", "correction", "correction_request"}:
        return "review"
    raise ValueError("Cannot infer handoff type. Set type=prototype_request/review_request or pass --action.")


def _as_pick_strings(items: list[Any], *, reject: bool) -> list[str]:
    result: list[str] = []
    for item in items:
        if isinstance(item, str):
            value = item.strip()
            if value:
                result.append(value)
            continue
        if not isinstance(item, dict):
            continue
        result_id = item.get("resultId", item.get("result_id"))
        if result_id is None:
            continue
        label = str(item.get("label") or ("reject" if reject else "")).strip()
        if not label:
            continue
        result.append(f"{int(result_id)}:{label}")
    return result


def apply_prototype(payload: dict[str, Any], *, chain: bool, seed: bool, policy: bool) -> int:
    db = str(payload.get("db") or payload.get("db_path") or "").strip()
    run_id = str(payload.get("run_id") or payload.get("runId") or "").strip()
    if not db or not run_id:
        raise ValueError("prototype request requires db/db_path and run_id")
    model_name = str(payload.get("model_name") or payload.get("modelName") or "facebook/dinov2-giant")
    view_name = str(payload.get("view_name") or payload.get("viewName") or "tight")
    notes = str(payload.get("notes") or f"handoff:{payload.get('request_path', '')}")
    prototypes = _as_pick_strings(list(payload.get("prototypes") or []), reject=False)
    rejects = _as_pick_strings(list(payload.get("rejects") or []), reject=True)
    argv = ["--db", db, "--run-id", run_id, "--model-name", model_name, "--view-name", view_name, "--notes", notes]
    if prototypes:
        argv.extend(["--prototype", ",".join(prototypes)])
    if rejects:
        argv.extend(["--reject", ",".join(rejects)])
    rc = _run_module("steps.step05_proto.main", argv)
    if rc != 0:
        return rc
    run_seed = bool(chain or seed or policy or payload.get("run_seed") or payload.get("runSeed"))
    run_policy = bool(chain or policy or payload.get("run_policy") or payload.get("runPolicy"))
    if run_seed:
        rc = _run_module("tools.relabel_semantic_seed", ["--db", db, "--run-id", run_id, "--apply"])
        if rc != 0:
            return rc
    if run_policy:
        rc = _run_module("steps.step06_reliability.main", ["--db", db, "--run-id", run_id])
        if rc != 0:
            return rc
        rc = _run_module("steps.step07_decision.main", ["--db", db, "--run-id", run_id])
    return rc


def apply_review(payload: dict[str, Any]) -> int:
    from semilabel_app.services.write_service import commit_corrections, commit_session

    db = str(payload.get("db") or payload.get("db_path") or "").strip()
    run_id = str(payload.get("run_id") or payload.get("runId") or "").strip()
    if not db or not run_id:
        raise ValueError("review request requires db/db_path and run_id")
    reviewer = str(payload.get("reviewer") or "")
    notes = str(payload.get("notes") or f"handoff:{payload.get('request_path', '')}")
    session_name = str(payload.get("session_name") or payload.get("sessionName") or "")
    decisions = list(payload.get("decisions") or [])
    corrections = list(payload.get("corrections") or [])
    results: dict[str, Any] = {}
    if decisions:
        results["decisions"] = commit_session(db, run_id, decisions, reviewer=reviewer, session_name=session_name, notes=notes)
    if corrections:
        results["corrections"] = commit_corrections(db, run_id, corrections, reviewer=reviewer, session_name=session_name, notes=notes)
    if not results:
        results["error"] = "No decisions/corrections in review request."
    print(json.dumps(results, ensure_ascii=False), flush=True)
    return 0 if not any(isinstance(item, dict) and item.get("error") for item in results.values()) else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = _read_request(args.request_json)
    action = _request_type(payload, str(args.action))
    if action == "prototype":
        return apply_prototype(payload, chain=bool(args.chain), seed=bool(args.seed), policy=bool(args.policy))
    if action == "review":
        return apply_review(payload)
    raise ValueError(f"Unknown handoff action: {action}")


if __name__ == "__main__":
    raise SystemExit(main())