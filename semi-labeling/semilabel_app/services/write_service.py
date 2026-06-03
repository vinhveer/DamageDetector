from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from .db_service import connect_rw


EXPORT_MAP = {
    "crack": "crack",
    "mold": "mold",
    "spall": "spall",
    "stain": "stain",
    "efflorescence": "stain",
    "shadow": "reject",
    "edge": "reject",
    "background": "reject",
    "object": "reject",
    "unknown": "reject",
    "reject": "reject",
}

VALID_ACTIONS = {"manual_accept", "manual_relabel", "manual_reject"}


def export_label(label: str) -> str:
    key = str(label or "").strip().lower()
    return EXPORT_MAP.get(key, "reject")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_stamped_id(name: str | None, fallback_prefix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name or "").strip()).strip("_")
    return f"{clean or fallback_prefix}_{stamp}"


def build_review_decisions(edits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for edit in edits:
        try:
            result_id = int(edit.get("resultId", edit.get("result_id")))
        except Exception:
            continue
        action = str(edit.get("action") or "").strip()
        if action not in VALID_ACTIONS:
            continue
        new_label = "reject" if action == "manual_reject" else str(edit.get("newLabel", edit.get("new_label")) or "").strip()
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


def commit_session(
    db_path: str,
    run_id: str,
    decisions: list[dict[str, Any]],
    reviewer: str = "",
    session_name: str = "",
    notes: str = "",
) -> dict[str, Any]:
    if not decisions:
        return {"error": "No decisions to commit."}
    session_id = make_stamped_id(session_name, "review")
    now = utc_now()
    conn = connect_rw(db_path)
    try:
        conn.execute("BEGIN")
        conn.execute(
            """
            INSERT INTO review_sessions (review_session_id, run_id, reviewer, status, created_at_utc, committed_at_utc, notes)
            VALUES (?, ?, ?, 'committed', ?, ?, ?)
            """,
            (session_id, run_id, reviewer, now, now, notes),
        )
        count = 0
        for decision in decisions:
            try:
                result_id = int(decision.get("resultId", decision.get("result_id")))
            except Exception:
                continue
            action = str(decision.get("action") or "").strip()
            if action not in VALID_ACTIONS:
                continue
            new_label = "reject" if action == "manual_reject" else str(decision.get("newLabel", decision.get("new_label")) or "").strip()
            conn.execute(
                """
                INSERT INTO review_decisions (
                  review_session_id, target_type, target_id, result_id, action,
                  previous_label, new_label, new_decision_type, reason_codes_json,
                  affected_result_ids_json, note, created_at_utc
                ) VALUES (?, 'result', ?, ?, ?, ?, ?, ?, '[]', ?, ?, ?)
                """,
                (
                    session_id,
                    str(result_id),
                    result_id,
                    action,
                    str(decision.get("previousLabel", decision.get("previous_label")) or ""),
                    new_label,
                    "reject" if action == "manual_reject" else "manual_accept",
                    json.dumps([result_id]),
                    str(decision.get("note") or ""),
                    now,
                ),
            )
            count += 1
        conn.commit()
        return {"committed": True, "reviewSessionId": session_id, "decisionCount": count, "committedAtUtc": now}
    except Exception as exc:
        conn.rollback()
        return {"error": str(exc)}
    finally:
        conn.close()


def update_cleaned_label(db_path: str, run_id: str, result_id: int, new_label: str) -> dict[str, Any]:
    final_label = str(new_label or "").strip().lower()
    if not final_label:
        return {"error": "Invalid newLabel"}
    conn = connect_rw(db_path)
    try:
        cur = conn.execute(
            """
            UPDATE cleaned_labels SET final_label = ?, export_label = ?
            WHERE run_id = ? AND result_id = ?
            """,
            (final_label, export_label(final_label), run_id, int(result_id)),
        )
        conn.commit()
        if not cur.rowcount:
            return {"updated": False, "error": "Row not found"}
        return {"updated": True, "finalLabel": final_label, "exportLabel": export_label(final_label)}
    except Exception as exc:
        conn.rollback()
        return {"error": str(exc)}
    finally:
        conn.close()


def commit_corrections(
    db_path: str,
    run_id: str,
    edits: list[dict[str, Any]],
    reviewer: str = "",
    session_name: str = "",
    notes: str = "",
) -> dict[str, Any]:
    rows = build_review_decisions(edits)
    if not rows:
        return {"error": "Chua co chinh sua nao"}
    session_id = make_stamped_id(session_name, "corrections")
    now = utc_now()
    conn = connect_rw(db_path)
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
        conn.commit()
        return {"committed": True, "reviewSessionId": session_id, "decisionCount": len(rows), "committedAtUtc": now}
    except Exception as exc:
        conn.rollback()
        return {"error": str(exc)}
    finally:
        conn.close()
