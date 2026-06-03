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
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.runtime import bootstrap

bootstrap.ensure_repo_root_on_path()


EXPORT_MAP = {
    "crack": "crack",
    "mold": "mold",
    "spall": "spall",
    "reject": "reject",
    "other": "reject",
    "stain": "reject",
    "efflorescence": "reject",
    "shadow": "reject",
    "edge": "reject",
    "background": "reject",
    "object": "reject",
    "unknown": "reject",
}
VALID_ACTIONS = {"manual_accept", "manual_relabel", "manual_reject"}


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


def _connect_rw(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {path}")
    conn = sqlite3.connect(str(path), timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _make_stamped_id(name: str | None, fallback_prefix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name or "").strip()).strip("_")
    return f"{clean or fallback_prefix}_{stamp}"


def _export_label(label: str) -> str:
    return EXPORT_MAP.get(str(label or "").strip().lower(), "reject")


def _normalise_decisions(edits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for edit in edits:
        try:
            result_id = int(edit.get("resultId", edit.get("result_id")))
        except Exception:
            continue
        action = str(edit.get("action") or "").strip()
        if action not in VALID_ACTIONS:
            continue
        new_label = "reject" if action == "manual_reject" else str(edit.get("newLabel", edit.get("new_label")) or "").strip().lower()
        if not new_label:
            continue
        rows.append(
            {
                "targetType": "result",
                "targetId": str(result_id),
                "resultId": result_id,
                "action": action,
                "previousLabel": str(edit.get("previousLabel", edit.get("previous_label")) or ""),
                "newLabel": new_label,
                "newDecisionType": "reject" if action == "manual_reject" else "manual_accept",
                "reasonCodesJson": json.dumps(["human_correction"]),
                "affectedResultIdsJson": json.dumps([result_id]),
                "note": str(edit.get("note") or ""),
            }
        )
    return rows


def _insert_review_rows(
    db_path: str,
    run_id: str,
    edits: list[dict[str, Any]],
    *,
    reviewer: str = "",
    session_name: str = "",
    notes: str = "",
    apply_to_cleaned: bool = False,
) -> dict[str, Any]:
    rows = _normalise_decisions(edits)
    if not rows:
        return {"error": "No valid review decisions."}
    session_id = _make_stamped_id(session_name, "review")
    now = _utc_now()
    conn = _connect_rw(db_path)
    try:
        conn.execute("BEGIN")
        conn.execute(
            """
            INSERT INTO review_sessions (review_session_id, run_id, reviewer, status, created_at_utc, committed_at_utc, notes)
            VALUES (?, ?, ?, 'committed', ?, ?, ?)
            """,
            (session_id, run_id, reviewer, now, now, notes),
        )
        for row in rows:
            conn.execute(
                """
                INSERT INTO review_decisions (
                  review_session_id, target_type, target_id, result_id, action,
                  previous_label, new_label, new_decision_type, reason_codes_json,
                  affected_result_ids_json, note, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    row["targetType"],
                    row["targetId"],
                    row["resultId"],
                    row["action"],
                    row["previousLabel"],
                    row["newLabel"],
                    row["newDecisionType"],
                    row["reasonCodesJson"],
                    row["affectedResultIdsJson"],
                    row["note"],
                    now,
                ),
            )
            if apply_to_cleaned:
                conn.execute(
                    """
                    UPDATE cleaned_labels
                    SET final_label = ?, export_label = ?, decision_type = ?
                    WHERE run_id = ? AND result_id = ?
                    """,
                    (row["newLabel"], _export_label(row["newLabel"]), row["newDecisionType"], run_id, row["resultId"]),
                )
                conn.execute(
                    """
                    UPDATE semantic_decisions
                    SET final_label = ?, decision_type = ?
                    WHERE run_id = ? AND result_id = ?
                    """,
                    (row["newLabel"], row["newDecisionType"], run_id, row["resultId"]),
                )
        conn.commit()
        return {"committed": True, "reviewSessionId": session_id, "decisionCount": len(rows), "committedAtUtc": now}
    except Exception as exc:
        conn.rollback()
        return {"error": str(exc)}
    finally:
        conn.close()


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
        results["decisions"] = _insert_review_rows(
            db, run_id, decisions, reviewer=reviewer, session_name=session_name, notes=notes, apply_to_cleaned=False
        )
    if corrections:
        results["corrections"] = _insert_review_rows(
            db, run_id, corrections, reviewer=reviewer, session_name=session_name, notes=notes, apply_to_cleaned=True
        )
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