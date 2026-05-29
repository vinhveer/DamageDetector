#!/usr/bin/env python3
"""Commit a human review session (SPEC 11) into the resemi SQLite DB.

Reads a JSON payload describing one drafted review session and writes it
transactionally into `review_sessions` / `review_decisions`. Prototype-target
decisions additionally produce a `prototype_versions` + `prototype_items` set.

Invoked by the Electron review console as a subprocess:

    python run_review_commit.py --input <payload.json>

The payload shape (produced by app/electron/review_console/sessions.js):

    {
      "output_db": "/abs/path/resemi.sqlite3",
      "session": {
        "session_id": "...",
        "run_id": "...",
        "reviewer": "...",
        "source_reliability_run_id": "...",
        "source_decision_policy_run_id": "...",
        "source_prototype_version_id": "...",
        "notes": "...",
        "decisions": { "<key>": { ...decision... }, ... }
      }
    }

Prints exactly one JSON result line to stdout for the Electron parser.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path


def _prepare_imports() -> None:
    package_parent = Path(__file__).resolve().parents[1]
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))


_prepare_imports()

from resemi.schema import connect_output, utc_now  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Commit a resemi human review session to SQLite.")
    parser.add_argument("--input", required=True, help="Path to JSON payload {output_db, session}.")
    return parser


def _as_int_or_none(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_list(value: object) -> str:
    if value is None:
        return "[]"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def commit_session(payload: dict) -> dict:
    output_db = str(payload.get("output_db") or "").strip()
    if not output_db:
        raise ValueError("Missing output_db")
    session = payload.get("session")
    if not isinstance(session, dict):
        raise ValueError("Missing session object")

    review_session_id = str(session.get("session_id") or "").strip()
    if not review_session_id:
        raise ValueError("Missing session.session_id")
    run_id = str(session.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("Missing session.run_id")

    decisions = session.get("decisions") or {}
    if not isinstance(decisions, dict):
        raise ValueError("session.decisions must be an object")

    created_at = utc_now()
    conn = connect_output(Path(output_db))
    try:
        existing = conn.execute(
            "SELECT status FROM review_sessions WHERE review_session_id = ?",
            (review_session_id,),
        ).fetchone()
        if existing is not None:
            raise RuntimeError(
                f"Review session already committed: {review_session_id}. "
                "Create a superseding session instead of overwriting."
            )

        conn.execute("BEGIN")

        conn.execute(
            """
            INSERT INTO review_sessions (
                review_session_id, run_id, reviewer, status,
                created_at_utc, committed_at_utc,
                source_reliability_run_id, source_decision_policy_run_id,
                source_prototype_version_id, notes
            ) VALUES (?, ?, ?, 'committed', ?, ?, ?, ?, ?, ?)
            """,
            (
                review_session_id,
                run_id,
                str(session.get("reviewer") or ""),
                str(session.get("created_at_utc") or created_at),
                created_at,
                session.get("source_reliability_run_id"),
                session.get("source_decision_policy_run_id"),
                session.get("source_prototype_version_id"),
                str(session.get("notes") or ""),
            ),
        )

        decision_rows = []
        prototype_items: list[dict] = []
        for key, dec in decisions.items():
            if not isinstance(dec, dict):
                continue
            target_type = str(dec.get("target_type") or "result").strip()
            target_id = str(dec.get("target_id") or key).strip()
            action = str(dec.get("action") or "").strip()
            if not action:
                continue
            result_id = _as_int_or_none(dec.get("result_id"))
            affected = dec.get("affected_result_ids")
            if not affected and result_id is not None:
                affected = [result_id]
            decision_rows.append(
                (
                    review_session_id,
                    target_type,
                    target_id,
                    result_id,
                    dec.get("core_cluster_id"),
                    dec.get("batch_id"),
                    action,
                    dec.get("previous_label"),
                    dec.get("new_label"),
                    dec.get("previous_decision_type"),
                    dec.get("new_decision_type"),
                    _as_float_or_none(dec.get("confidence_override")),
                    _json_list(dec.get("reason_codes")),
                    _json_list(affected),
                    str(dec.get("note") or ""),
                    str(dec.get("decided_at_utc") or created_at),
                )
            )
            if target_type == "prototype" and result_id is not None and action in {"add_prototype", "add_reject_prototype"}:
                prototype_items.append(
                    {
                        "result_id": result_id,
                        "label": str(dec.get("new_label") or dec.get("previous_label") or ""),
                        "is_reject": 1 if action == "add_reject_prototype" else 0,
                        "note": str(dec.get("note") or ""),
                    }
                )

        conn.executemany(
            """
            INSERT INTO review_decisions (
                review_session_id, target_type, target_id, result_id,
                core_cluster_id, batch_id, action,
                previous_label, new_label, previous_decision_type, new_decision_type,
                confidence_override, reason_codes_json, affected_result_ids_json,
                note, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            decision_rows,
        )

        prototype_version_id = None
        if prototype_items:
            prototype_version_id = f"protov_{uuid.uuid4().hex[:12]}"
            label_map: dict[str, list[int]] = {}
            for item in prototype_items:
                label_map.setdefault(item["label"], []).append(item["result_id"])
            conn.execute(
                """
                INSERT INTO prototype_versions (
                    prototype_version_id, run_id, created_at_utc, notes, source_session,
                    label_map_json, selected_result_ids_json, selected_cluster_ids_json,
                    excluded_ids_json, model_name, view_name, options_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '', '', '{}')
                """,
                (
                    prototype_version_id,
                    run_id,
                    created_at,
                    f"Picked in review session {review_session_id}",
                    review_session_id,
                    json.dumps(label_map, ensure_ascii=False, sort_keys=True),
                    json.dumps([it["result_id"] for it in prototype_items], ensure_ascii=False),
                    "[]",
                    "[]",
                ),
            )
            conn.executemany(
                """
                INSERT INTO prototype_items (
                    prototype_version_id, result_id, label, is_reject, note,
                    source_type, source_ref, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, 'review_pick', ?, ?)
                """,
                [
                    (
                        prototype_version_id,
                        it["result_id"],
                        it["label"],
                        it["is_reject"],
                        it["note"],
                        review_session_id,
                        created_at,
                    )
                    for it in prototype_items
                ],
            )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return {
        "committed": True,
        "review_session_id": review_session_id,
        "run_id": run_id,
        "decision_count": len(decision_rows),
        "prototype_version_id": prototype_version_id,
        "committed_at_utc": created_at,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        result = commit_session(payload)
    except Exception as exc:  # noqa: BLE001 - surface any failure as JSON for the UI
        print(json.dumps({"error": str(exc)}), flush=True)
        return 1
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
