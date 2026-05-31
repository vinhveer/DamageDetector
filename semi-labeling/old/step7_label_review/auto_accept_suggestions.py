#!/usr/bin/env python3
"""Auto-accept every cluster suggestion from cluster_suspects.py output.

Writes a session JSON (decisions keyed by result_id) so the UI can still open
it for spot-check; then optionally runs apply_decisions.py to emit a final
labels CSV in one go.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def resolve_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "object_detection").exists() and (candidate / "tools").exists():
            return candidate
    return current.parents[2]


REPO_ROOT = resolve_repo_root()
LAB_ROOT = REPO_ROOT.parent


def default_step7_dir() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "step7_label_review"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def connect_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).expanduser().resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=60.0)
    conn.row_factory = sqlite3.Row
    return conn


def pick_latest_run(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT run_id FROM suspect_cluster_runs ORDER BY created_at_utc DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("No suspect_cluster_runs found.")
    return str(row["run_id"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auto-accept every suspect-cluster suggestion.")
    parser.add_argument("--suspect-cluster-db", default=str(default_step7_dir() / "suspect_clusters.sqlite3"))
    parser.add_argument("--run-id", default=None, help="Run id. Default: latest.")
    parser.add_argument("--title", default=None, help="Session title (default: 'auto-accept <run_id>')")
    parser.add_argument("--step7-dir", default=str(default_step7_dir()))
    parser.add_argument("--include-noise", action="store_true",
                        help="Also auto-accept noise groups (default: skip noise — they need manual review).")
    args = parser.parse_args(argv)

    step7_dir = Path(args.step7_dir).expanduser().resolve()
    db_path = Path(args.suspect_cluster_db).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Suspect cluster DB not found: {db_path}")

    conn = connect_ro(db_path)
    try:
        run_id = args.run_id or pick_latest_run(conn)
        print(f"[input] run_id = {run_id}")
        summaries = conn.execute(
            """
            SELECT current_label, cluster_id, size, dominant_cv_label, dominant_cv_fraction,
                   suggested_action, suggested_target_label, is_noise_cluster
            FROM suspect_cluster_summary
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall()
        members = conn.execute(
            """
            SELECT current_label, cluster_id, result_id, cv_predicted_label, suspicion_score
            FROM suspect_cluster_results
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall()
    finally:
        conn.close()

    # Index members by (current_label, cluster_id)
    members_by_group: dict[tuple[str, int], list[dict]] = {}
    for m in members:
        key = (str(m["current_label"]), int(m["cluster_id"]))
        members_by_group.setdefault(key, []).append(dict(m))

    ts = utc_now()
    decisions: dict[str, dict] = {}
    action_counter: Counter = Counter()
    transition_counter: Counter = Counter()

    for s in summaries:
        cls = str(s["current_label"])
        sub_id = int(s["cluster_id"])
        action = str(s["suggested_action"])
        target = str(s["suggested_target_label"])
        is_noise = bool(s["is_noise_cluster"])
        if is_noise and not args.include_noise:
            action_counter["skipped_noise"] += 1
            continue

        for m in members_by_group.get((cls, sub_id), []):
            rid = int(m["result_id"])
            # Prefer the cluster's suggestion. For noise groups (when --include-noise),
            # fall back to the per-box CV prediction since the cluster suggestion is weak.
            box_action = action
            box_target = target
            if is_noise and args.include_noise:
                cv_pred = str(m.get("cv_predicted_label") or "")
                if cv_pred and cv_pred != cls:
                    box_action = "change"
                    box_target = cv_pred
                else:
                    box_action = "keep"
                    box_target = cls

            if box_action == "keep":
                decisions[str(rid)] = {
                    "action": "keep",
                    "target_label": None,
                    "current_label_at_decision": cls,
                    "source": "auto_accept_suggestion",
                    "decided_at_utc": ts,
                }
                action_counter["keep"] += 1
            elif box_action == "change" and box_target:
                decisions[str(rid)] = {
                    "action": "change",
                    "target_label": box_target,
                    "current_label_at_decision": cls,
                    "source": "auto_accept_suggestion",
                    "decided_at_utc": ts,
                }
                action_counter["change"] += 1
                transition_counter[f"{cls} → {box_target}"] += 1
            else:
                action_counter["skipped_unknown"] += 1

    # Build session JSON
    sessions_dir = step7_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex[:16]
    stats = {
        "reviewed": len(decisions),
        "kept": action_counter["keep"],
        "changed": action_counter["change"],
        "rejected": sum(1 for d in decisions.values()
                        if d["action"] == "change" and d["target_label"] == "reject"),
    }
    session_payload = {
        "session_id": session_id,
        "subcluster_run_id": run_id,
        "title": args.title or f"auto-accept {run_id[:8]}",
        "created_at_utc": ts,
        "last_updated_utc": ts,
        "decisions": decisions,
        "stats": stats,
    }
    session_path = sessions_dir / f"{session_id}.json"
    session_path.write_text(json.dumps(session_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[summary]")
    for k, v in action_counter.most_common():
        print(f"  {k:<24} {v:>6}")
    if transition_counter:
        print(f"\n[top transitions]")
        for trans, n in transition_counter.most_common(12):
            print(f"  {trans:<28} {n:>6}")
    print(f"\n[saved session] {session_path}")
    print(f"  · session_id  = {session_id}")
    print(f"  · decisions   = {len(decisions)}")
    print(f"\nNext: chạy apply_decisions để emit CSV cuối:")
    print(f"  python -m DamageDetector.semi-labeling.step7_label_review.apply_decisions \\")
    print(f"      --session {session_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
