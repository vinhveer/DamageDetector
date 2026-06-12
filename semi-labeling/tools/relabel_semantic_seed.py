#!/usr/bin/env python3
"""Relabel semantic seed labels from detector + DINOv2 evidence.

This tool intentionally does not run step01. It updates only the existing
semantic_decisions seed labels so cached crop embeddings remain intact.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from shared.runtime import bootstrap

bootstrap.ensure_on_path()

from shared.db.schema import connect_output, utc_now  # noqa: E402
from shared.runtime.paths import default_resemi_db  # noqa: E402


DAMAGE_LABELS = ("crack", "mold", "spall")
REJECT_LABELS = {"reject", "other", "unknown", "background", "shadow", "edge", "object"}
DYNAMIC_REASON_CODES = {
    "high_consensus",
    "low_margin",
    "low_semantic_reliability",
    "needs_core_or_prototype_evidence",
    "missing_core_signal",
    "missing_prototype_signal",
    "prototype_label_disagree",
    "openclip_core_disagree",
    "near_reject_prototype",
    "far_from_class_core",
    "multi_crop_inconsistent",
    "geometry_conflict",
    "rare_cluster",
    "noise_cluster",
    "insufficient_class_samples",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Relabel semantic_decisions using 2-of-3 GDINO/prototype/core voting without deleting embeddings."
    )
    parser.add_argument("--db", default=str(default_resemi_db()), help="Resemi SQLite DB.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-db", default="", help="Override source pipeline SQLite DB. Default: resemi_runs.source_db_path.")
    parser.add_argument("--source-semantic-run-id", default="", help="Override source semantic run id. Default: resemi_runs.source_semantic_run_id.")
    parser.add_argument("--min-votes", type=int, default=2, help="Votes required among detector/prototype/core.")
    parser.add_argument("--labels", nargs="*", default=list(DAMAGE_LABELS), help="Damage labels eligible for seed voting.")
    parser.add_argument("--reject-prototype-sim", type=float, default=0.50,
                        help="Reject when a manual reject prototype wins at or above this similarity and damage vote is below --min-votes.")
    parser.add_argument("--openclip-other-reject", action=argparse.BooleanOptionalAction, default=False,
                        help="Reject OpenCLIP 'other'/reject seeds when damage vote is below --min-votes. Off by default because OpenCLIP is auxiliary.")
    parser.add_argument("--apply", action="store_true", help="Write changes. Omitted means dry-run.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = Path(args.db).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Resemi DB not found: {db_path}")
    labels = tuple(str(item).strip().lower() for item in args.labels if str(item).strip())
    if not labels:
        raise ValueError("At least one label is required.")
    if int(args.min_votes) < 1:
        raise ValueError("--min-votes must be >= 1.")

    conn = connect_output(db_path)
    try:
        run = _read_run(conn, run_id=str(args.run_id))
        source_db = Path(args.source_db or run["source_db_path"]).expanduser().resolve()
        semantic_run_id = str(args.source_semantic_run_id or run["source_semantic_run_id"])
        if not source_db.is_file():
            raise FileNotFoundError(f"Source DB not found: {source_db}")
        conn.execute("ATTACH DATABASE ? AS src", (str(source_db),))
        rows = _read_rows(conn, run_id=str(args.run_id), semantic_run_id=semantic_run_id)
        updates, summary = _build_updates(
            rows,
            labels=labels,
            min_votes=int(args.min_votes),
            reject_prototype_sim=float(args.reject_prototype_sim),
            openclip_other_reject=bool(args.openclip_other_reject),
        )
        _print_summary(
            run_id=str(args.run_id),
            db_path=db_path,
            source_db=source_db,
            semantic_run_id=semantic_run_id,
            summary=summary,
            dry_run=not bool(args.apply),
        )
        if not args.apply:
            return 0
        _persist(conn, run_id=str(args.run_id), updates=updates)
        print(f"updated_count={len(updates)}")
        print("apply=True")
        return 0
    except sqlite3.Error:
        conn.rollback()
        raise
    finally:
        conn.close()


def _read_run(conn: sqlite3.Connection, *, run_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT run_id, source_db_path, source_semantic_run_id FROM resemi_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Run not found: {run_id}")
    return row


def _read_rows(conn: sqlite3.Connection, *, run_id: str, semantic_run_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT d.run_id, d.result_id, d.initial_label, d.suggested_label, d.final_label,
               d.decision_type, d.reliability_score, d.reason_codes_json, d.score_components_json,
               d.prototype_class, d.prototype_similarity, d.nearest_core_class, d.nearest_core_similarity,
               r.label AS detector_label, r.prompt_key, r.score AS detector_score
        FROM semantic_decisions d
        JOIN src.detections r
          ON r.detection_id = d.result_id AND r.run_id = ?
        WHERE d.run_id = ?
        ORDER BY d.result_id
        """,
        (semantic_run_id, run_id),
    ).fetchall()


def _build_updates(
    rows: list[sqlite3.Row],
    *,
    labels: tuple[str, ...],
    min_votes: int,
    reject_prototype_sim: float,
    openclip_other_reject: bool,
) -> tuple[list[tuple], dict[str, object]]:
    eligible = set(labels)
    updates: list[tuple] = []
    before = Counter()
    after = Counter()
    changed = Counter()
    vote_patterns = Counter()
    now = utc_now()
    for row in rows:
        current = str(row["final_label"])
        before[current] += 1
        votes = {
            "detector": _normalize_detector_label(f"{row['detector_label']} {row['prompt_key']}"),
            "prototype": _normalize_label(row["prototype_class"]),
            "core": _normalize_label(row["nearest_core_class"]),
        }
        components = _parse_json_dict(row["score_components_json"])
        chosen, count = _choose_label(votes, eligible=eligible, min_votes=min_votes)
        chosen_source = "detector_prototype_core_vote"
        if chosen is None:
            reject_reason = _choose_reject(
                row,
                components,
                votes=votes,
                eligible=eligible,
                damage_vote_count=count,
                min_votes=min_votes,
                reject_prototype_sim=reject_prototype_sim,
                openclip_other_reject=openclip_other_reject,
            )
            if reject_reason:
                chosen = "reject"
                chosen_source = reject_reason
        if chosen is None:
            after[current] += 1
            continue
        after[chosen] += 1
        vote_patterns[(votes["detector"], votes["prototype"], votes["core"], chosen)] += 1
        if chosen == current:
            continue
        reasons = [
            item
            for item in _parse_json_list(row["reason_codes_json"])
            if item not in DYNAMIC_REASON_CODES
        ]
        if chosen == "reject":
            reasons.append("seed_reject_other_or_reject_prototype")
        else:
            reasons.append("seed_relabel_detector_dinov2_vote")
        reasons.append(f"seed_vote_count_{count}")
        openclip_top_label = str(components.get("top_label", row["suggested_label"]))
        components.update(
            {
                "openclip_top_label": openclip_top_label,
                "seed_label_source": chosen_source,
                "seed_label_votes": votes,
                "seed_vote_count": int(count),
                "top_label": chosen,
            }
        )
        updates.append(
            (
                chosen,
                chosen,
                "reject" if chosen == "reject" else "suspect",
                json.dumps(sorted(set(reasons)), ensure_ascii=False, sort_keys=True),
                json.dumps(components, ensure_ascii=False, sort_keys=True),
                now,
                int(row["result_id"]),
            )
        )
        changed[(current, chosen)] += 1
    summary = {
        "total": len(rows),
        "before": dict(sorted(before.items())),
        "after_if_applied": dict(sorted(after.items())),
        "changed_count": len(updates),
        "changed": {f"{src}->{dst}": count for (src, dst), count in sorted(changed.items())},
        "top_vote_patterns": [
            {"detector": key[0], "prototype": key[1], "core": key[2], "chosen": key[3], "count": count}
            for key, count in vote_patterns.most_common(15)
        ],
    }
    return updates, summary


def _choose_reject(
    row: sqlite3.Row,
    components: dict[str, object],
    *,
    votes: dict[str, str | None],
    eligible: set[str],
    damage_vote_count: int,
    min_votes: int,
    reject_prototype_sim: float,
    openclip_other_reject: bool,
) -> str | None:
    if damage_vote_count >= min_votes:
        return None
    prototype_label = _normalize_label(row["prototype_class"])
    prototype_sim = _safe_float(row["prototype_similarity"])
    if prototype_label in REJECT_LABELS and prototype_sim is not None and prototype_sim >= reject_prototype_sim:
        return "reject_prototype"
    if not openclip_other_reject:
        return None
    openclip_label = _normalize_label(
        components.get("openclip_top_label")
        or components.get("top_label")
        or row["initial_label"]
        or row["suggested_label"]
    )
    if openclip_label in REJECT_LABELS:
        damage_votes = sum(1 for label in votes.values() if label in eligible)
        if damage_votes < min_votes:
            return "openclip_other"
    return None


def _persist(conn: sqlite3.Connection, *, run_id: str, updates: list[tuple]) -> None:
    backup_table = f"semantic_seed_relabel_backup_{_safe_identifier(run_id)}"
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {backup_table} AS
        SELECT run_id, result_id, initial_label, suggested_label, final_label, decision_type,
               reliability_score, reason_codes_json, score_components_json, created_at_utc
        FROM semantic_decisions
        WHERE 0
        """
    )
    conn.execute(f"DELETE FROM {backup_table} WHERE run_id = ?", (run_id,))
    conn.execute(
        f"""
        INSERT INTO {backup_table}
        SELECT run_id, result_id, initial_label, suggested_label, final_label, decision_type,
               reliability_score, reason_codes_json, score_components_json, created_at_utc
        FROM semantic_decisions
        WHERE run_id = ?
        """,
        (run_id,),
    )
    conn.executemany(
        """
        UPDATE semantic_decisions
        SET suggested_label = ?, final_label = ?, decision_type = ?,
            reason_codes_json = ?, score_components_json = ?, created_at_utc = ?,
            matched_rule = NULL
        WHERE run_id = ? AND result_id = ?
        """,
        [(suggested, final, dtype, reasons, components, created_at, run_id, result_id) for suggested, final, dtype, reasons, components, created_at, result_id in updates],
    )
    conn.commit()


def _choose_label(votes: dict[str, str | None], *, eligible: set[str], min_votes: int) -> tuple[str | None, int]:
    counts = Counter(label for label in votes.values() if label in eligible)
    if not counts:
        return None, 0
    label, count = counts.most_common(1)[0]
    if count < min_votes:
        return None, count
    tied = [item for item, item_count in counts.items() if item_count == count]
    if len(tied) > 1:
        return None, count
    return label, count


def _normalize_label(value: object) -> str | None:
    label = str(value or "").strip().lower()
    if not label or label in REJECT_LABELS:
        return label or None
    return label


def _normalize_detector_label(value: object) -> str | None:
    text = str(value or "").lower()
    aliases = {
        "spall": ("spall", "spalling", "delamination", "flaking"),
        "mold": ("mold", "mould", "stain", "moisture", "dirty"),
        "crack": ("crack", "fracture", "fissure"),
    }
    for label, terms in aliases.items():
        if any(term in text for term in terms):
            return label
    return None


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_json_list(raw: object) -> list[str]:
    try:
        value = json.loads(str(raw or "[]"))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


def _parse_json_dict(raw: object) -> dict[str, object]:
    try:
        value = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _safe_identifier(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value)


def _print_summary(
    *,
    run_id: str,
    db_path: Path,
    source_db: Path,
    semantic_run_id: str,
    summary: dict[str, object],
    dry_run: bool,
) -> None:
    print(f"run_id={run_id}")
    print(f"db={db_path}")
    print(f"source_db={source_db}")
    print(f"source_semantic_run_id={semantic_run_id}")
    print(f"dry_run={dry_run}")
    print(f"total={summary['total']}")
    print(f"changed_count={summary['changed_count']}")
    print("before=" + json.dumps(summary["before"], sort_keys=True))
    print("after_if_applied=" + json.dumps(summary["after_if_applied"], sort_keys=True))
    print("changed=" + json.dumps(summary["changed"], sort_keys=True))
    print("top_vote_patterns=" + json.dumps(summary["top_vote_patterns"], sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
