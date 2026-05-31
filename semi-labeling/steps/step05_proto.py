#!/usr/bin/env python3
"""Step 05 — Prototype Bank.

Creates and scores a prototype bank from human-picked detections (or core
clusters). Optional. Run after step03/step04.

HUMAN GATE: prototypes are picked by a person (20-50/class + rejects) and
passed via --prototype / --reject / --cluster.

Inputs:  crop_embeddings (+ core clusters) in resemi.sqlite3
Outputs: prototype_versions, prototype_items, prototype_scoring_runs, prototype_scores
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from lib import bootstrap

bootstrap.ensure_on_path()

from lib.paths import default_resemi_db  # noqa: E402
from lib.prototype_bank import (  # noqa: E402
    PrototypeBankConfig,
    PrototypeSpec,
    build_prototype_bank,
    persist_prototype_bank,
    persist_prototype_scores,
    score_prototype_bank_preview,
    score_prototypes,
)
from lib.schema import connect_output, utc_now  # noqa: E402


DEFAULT_MODEL_NAME = "facebook/dinov2-small"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create and score a resemi prototype bank from human-selected detections or core clusters.")
    parser.add_argument("--db", default=str(default_resemi_db()), help="Resemi SQLite DB.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--view-name", default="tight")
    parser.add_argument("--embedding-run-id", default="")
    parser.add_argument("--prototype-version-id", default="")
    parser.add_argument("--source-session", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--prototype", action="append", default=[], help="Manual pick as result_id:label. Repeatable or comma-separated.")
    parser.add_argument("--reject", action="append", default=[], help="Reject pick as result_id[:label]. Label defaults to reject.")
    parser.add_argument("--cluster", action="append", default=[], help="Core cluster pick as core_cluster_id[:label].")
    parser.add_argument("--exclude-id", action="append", default=[], help="Exclude result_id. Repeatable or comma-separated.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--update-decisions", action=argparse.BooleanOptionalAction, default=True)
    return parser


def parse_ids(raw: list[str]) -> tuple[int, ...]:
    result: list[int] = []
    for item in raw:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                result.append(int(part))
    return tuple(result)


def parse_prototypes(raw: list[str], *, is_reject: bool) -> list[PrototypeSpec]:
    result: list[PrototypeSpec] = []
    for item in raw:
        for part in str(item).split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                result_id, label = part.split(":", 1)
                label = label.strip() or "reject"
            else:
                result_id, label = part, "reject" if is_reject else ""
            if not label:
                raise ValueError(f"Prototype label is required: {part}")
            result.append(PrototypeSpec(result_id=int(result_id), label=label, is_reject=is_reject, source_type="manual_reject" if is_reject else "manual_result"))
    return result


def parse_clusters(raw: list[str]) -> tuple[tuple[str, str | None], ...]:
    result: list[tuple[str, str | None]] = []
    for item in raw:
        for part in str(item).split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                cluster_id, label = part.split(":", 1)
                result.append((cluster_id.strip(), label.strip() or None))
            else:
                result.append((part, None))
    return tuple(result)


def print_summary(bank, scores) -> None:
    by_label: dict[str, int] = {}
    for proto in bank.prototypes:
        by_label[proto.label] = by_label.get(proto.label, 0) + 1
    print(f"prototype_version_id={bank.prototype_version_id}")
    print(f"prototype_score_run_id={scores.prototype_score_run_id}")
    print(f"embedding_run_id={bank.embedding_run_id}")
    print(f"prototype_count={len(bank.prototypes)}")
    print(f"scored_count={len(scores.scores)}")
    print("prototypes_by_label=" + ",".join(f"{label}:{count}" for label, count in sorted(by_label.items())))
    reject_matches = sum(1 for score in scores.scores if score.is_reject_match)
    low_margin = sum(1 for score in scores.scores if "prototype_low_margin" in score.reason_codes)
    print(f"reject_matches={reject_matches}")
    print(f"low_margin={low_margin}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = Path(args.db).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Resemi DB not found: {db_path}")
    manual = [*parse_prototypes(list(args.prototype or []), is_reject=False), *parse_prototypes(list(args.reject or []), is_reject=True)]
    config = PrototypeBankConfig(
        run_id=str(args.run_id),
        model_name=str(args.model_name),
        view_name=str(args.view_name),
        embedding_run_id=str(args.embedding_run_id or ""),
        prototype_version_id=str(args.prototype_version_id or ""),
        source_session=str(args.source_session or ""),
        notes=str(args.notes or ""),
        selected_clusters=parse_clusters(list(args.cluster or [])),
        excluded_ids=parse_ids(list(args.exclude_id or [])),
    )
    conn = connect_output(db_path)
    try:
        bank = build_prototype_bank(conn, config=config, manual_prototypes=manual)
        scores = score_prototype_bank_preview(conn, bank=bank, top_k=int(args.top_k)) if bool(args.dry_run) else None
        if not bool(args.dry_run):
            options = {
                "manual_prototypes": [f"{item.result_id}:{item.label}" for item in manual],
                "selected_clusters": [f"{cluster_id}:{label}" if label else cluster_id for cluster_id, label in config.selected_clusters],
                "excluded_ids": list(config.excluded_ids),
                "top_k": int(args.top_k),
                "update_decisions": bool(args.update_decisions),
            }
            now = utc_now()
            persist_prototype_bank(conn, bank, created_at_utc=now, source_session=config.source_session, notes=config.notes, options=options)
            scores = score_prototypes(
                conn,
                prototype_version_id=bank.prototype_version_id,
                run_id=bank.run_id,
                model_name=bank.model_name,
                view_name=bank.view_name,
                embedding_run_id=bank.embedding_run_id,
                top_k=int(args.top_k),
            )
            persist_prototype_scores(conn, scores, created_at_utc=utc_now(), update_decisions=bool(args.update_decisions))
        if scores is None:
            raise RuntimeError("Prototype scoring did not run.")
        print_summary(bank, scores)
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
