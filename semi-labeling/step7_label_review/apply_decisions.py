#!/usr/bin/env python3
"""Apply Step 7 per-box decisions onto complete_labels.csv → final_labels.csv.

Sessions store decisions keyed by `result_id`. Each decision is one of:
  - {action: 'keep'} → confirm current label
  - {action: 'change', target_label: 'mold'|'spall'|'crack'|'reject'} → relabel
The HDBSCAN sub-cluster structure is only used by the UI for browsing; it isn't
needed at apply time.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def resolve_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "object_detection").exists() and (candidate / "tools").exists():
            return candidate
    return current.parents[2]


REPO_ROOT = resolve_repo_root()
LAB_ROOT = REPO_ROOT.parent


def default_step6_dir() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "step6_classifier"


def default_step7_dir() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "step7_label_review"


def pick_latest(folder: Path, glob: str) -> Path:
    files = sorted(folder.glob(glob), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No files match {folder}/{glob}")
    return files[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply Step 7 per-box decisions.")
    parser.add_argument("--complete-labels-csv", default=None, help="Default: latest complete_labels_*.csv in step6 dir.")
    parser.add_argument("--session", required=True, help="Session JSON file (absolute or filename in step7 sessions dir).")
    parser.add_argument("--output", default=None, help="Output CSV path. Default: step7_dir/final_labels_<session_id>.csv")
    parser.add_argument("--step6-dir", default=str(default_step6_dir()))
    parser.add_argument("--step7-dir", default=str(default_step7_dir()))
    parser.add_argument("--corrections-db", default=None, help="If set, emit LabelCorrection rows into this corrections.sqlite3 (C6 feedback loop).")
    parser.add_argument("--source-run-id", default="", help="Source run id recorded on emitted corrections.")
    args = parser.parse_args(argv)

    step6_dir = Path(args.step6_dir).expanduser().resolve()
    step7_dir = Path(args.step7_dir).expanduser().resolve()

    csv_path = (
        Path(args.complete_labels_csv).expanduser().resolve()
        if args.complete_labels_csv
        else pick_latest(step6_dir, "complete_labels_*.csv")
    )

    session_arg = args.session
    session_path = Path(session_arg).expanduser()
    if not session_path.is_absolute():
        candidate = step7_dir / "sessions" / session_arg
        if not candidate.exists() and not session_arg.endswith(".json"):
            candidate = step7_dir / "sessions" / f"{session_arg}.json"
        session_path = candidate
    if not session_path.is_file():
        raise FileNotFoundError(f"Session not found: {args.session}")

    session = json.loads(session_path.read_text(encoding="utf-8"))
    raw_decisions = session.get("decisions", {})
    session_id = str(session.get("session_id", session_path.stem))

    # Decisions are keyed by result_id (per-box). Normalize keys to int.
    decisions: dict[int, dict] = {}
    for k, v in raw_decisions.items():
        try:
            decisions[int(k)] = v
        except (TypeError, ValueError):
            continue

    print(f"[input] csv = {csv_path.name}")
    print(f"[input] session = {session_path.name}  ({len(decisions)} per-box decisions)")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else step7_dir / f"final_labels_{session_id}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_in: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_in.append(row)

    counter: Counter = Counter()
    label_changes: Counter = Counter()
    pending_corrections: list[tuple] = []  # (rid, image_rel_path, original_label, correction_type, corrected_label)
    rows_out: list[dict] = []
    for row in rows_in:
        rid_str = str(row.get("result_id", ""))
        try:
            rid = int(rid_str)
        except ValueError:
            rid = None

        out = dict(row)
        decision = decisions.get(rid) if rid is not None else None
        if not decision or not isinstance(decision, dict):
            counter["untouched"] += 1
            rows_out.append(out)
            continue

        action = str(decision.get("action", "")).strip()
        old_label = str(row.get("final_class", "")).strip()
        image_rel_path = str(row.get("image_rel_path", ""))
        if action == "keep":
            out["source"] = "reviewed_keep"
            counter["reviewed_keep"] += 1
            pending_corrections.append((rid, image_rel_path, old_label, "confirm", old_label))
        elif action == "change":
            target = str(decision.get("target_label", "")).strip()
            if not target:
                counter["change_no_target"] += 1
            else:
                out["final_class"] = target
                out["source"] = "reviewed_reject" if target == "reject" else "reviewed_change"
                out["confidence"] = "1.0"
                if target == "reject":
                    counter["reviewed_reject"] += 1
                    pending_corrections.append((rid, image_rel_path, old_label, "reject", "reject"))
                else:
                    counter["reviewed_change"] += 1
                    pending_corrections.append((rid, image_rel_path, old_label, "relabel", target))
                if old_label and old_label != target:
                    label_changes[f"{old_label} → {target}"] += 1
        else:
            counter[f"unknown_action_{action}"] += 1
        rows_out.append(out)

    fieldnames = list(rows_in[0].keys()) if rows_in else [
        "result_id", "image_rel_path", "cluster_id", "predicted_label_step2", "final_class", "source", "confidence",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\n[summary]")
    for key, value in counter.most_common():
        print(f"  {key:<24} {value:>6}")
    if label_changes:
        print(f"\n[label changes]")
        for trans, n in label_changes.most_common():
            print(f"  {trans:<28} {n:>6}")

    final_dist = Counter(r["final_class"] for r in rows_out if r.get("final_class"))
    print(f"\n[final distribution]")
    for label, n in sorted(final_dist.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<10} {n:>6}")

    print(f"\n[saved] {output_path}")

    if args.corrections_db and pending_corrections:
        from datetime import datetime, timezone

        from corrections.store import CorrectionStore, CorrectionWriteError, LabelCorrection

        store = CorrectionStore(Path(args.corrections_db).expanduser().resolve())
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        written = 0
        for rid, image_rel_path, original_label, ctype, corrected_label in pending_corrections:
            try:
                store.add_correction(
                    LabelCorrection(
                        correction_id=f"{session_id}:{rid}",
                        created_at_utc=now,
                        source_run_id=str(args.source_run_id),
                        result_id=int(rid),
                        image_rel_path=image_rel_path,
                        original_label=original_label,
                        corrected_label=corrected_label,
                        correction_type=ctype,
                    )
                )
                written += 1
            except CorrectionWriteError as exc:
                print(f"  [correction skipped] {exc.correction_id}: {exc}")
        print(f"[corrections] wrote {written}/{len(pending_corrections)} into {args.corrections_db}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
