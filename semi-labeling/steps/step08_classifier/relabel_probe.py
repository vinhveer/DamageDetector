#!/usr/bin/env python3
"""Re-label review_queue boxes with a DINOv2 linear probe.

The diagnostic probe showed DINOv2-giant embeddings separate crack/mold/spall
well (~96%), while the pipeline's nearest-prototype voting is the weak link.
This tool trains a logistic-regression probe on the current cleaned_labels and
predicts labels for boxes still sitting in review_queue, then (optionally)
promotes the confident ones into cleaned_labels.

Default is a dry-run that reports how many boxes each probability threshold
would promote. Writing requires both --apply and --min-prob.

Usage:
    # report only (no DB write)
    python -m steps.step08_classifier.relabel_probe --db DB --run-id myrun --include-reject
    # promote boxes with probe prob >= 0.90
    python -m steps.step08_classifier.relabel_probe --db DB --run-id myrun --include-reject --min-prob 0.90 --apply
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from shared.runtime import bootstrap

bootstrap.ensure_on_path()

from shared.db.embedding_cache import load_embeddings  # noqa: E402
from shared.db.schema import connect_output  # noqa: E402

DEFAULT_MODEL_NAME = "facebook/dinov2-giant"
DAMAGE_LABELS = ("crack", "mold", "spall")
REPORT_THRESHOLDS = (0.70, 0.80, 0.90, 0.95)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-label review_queue with a DINOv2 linear probe.")
    parser.add_argument("--db", required=True, help="Resemi SQLite DB.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--view-name", default="tight")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularisation for LogisticRegression.")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                        help="L2-normalise embeddings before probing. Default on.")
    parser.add_argument("--include-reject", action=argparse.BooleanOptionalAction, default=True,
                        help="Re-examine queue_type='reject' boxes too. Default on.")
    parser.add_argument("--min-prob", type=float, default=0.0,
                        help="Promote review boxes with max probe probability >= this. 0 = report only.")
    parser.add_argument("--out-db", default="",
                        help="Write a standalone dataset SQLite (original cleaned + probe promotes) here. "
                             "Source --db is read-only. Requires --min-prob > 0.")
    return parser


def _read_cleaned_labels(conn, *, run_id: str) -> dict[int, str]:
    rows = conn.execute(
        "SELECT result_id, final_label FROM cleaned_labels WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    out: dict[int, str] = {}
    for row in rows:
        label = str(row["final_label"] or "").strip().lower()
        if label in DAMAGE_LABELS:
            out[int(row["result_id"])] = label
    return out


def _read_review_queue(conn, *, run_id: str, include_reject: bool) -> dict[int, str]:
    rows = conn.execute(
        "SELECT result_id, queue_type FROM review_queue WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    out: dict[int, str] = {}
    for row in rows:
        qtype = str(row["queue_type"] or "")
        if not include_reject and qtype == "reject":
            continue
        out[int(row["result_id"])] = qtype
    return out


def main(argv: list[str] | None = None) -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    args = build_parser().parse_args(argv)
    if args.out_db and float(args.min_prob) <= 0:
        raise SystemExit("--out-db requires --min-prob > 0 (nothing to promote at 0).")

    conn = connect_output(args.db)
    try:
        run_id = str(args.run_id)
        model_name = str(args.model_name)
        view_name = str(args.view_name)

        # 1. Train set = current cleaned damage labels.
        train_labels = _read_cleaned_labels(conn, run_id=run_id)
        if not train_labels:
            raise SystemExit(f"No cleaned damage labels for run_id={run_id!r}.")
        train_ids = sorted(train_labels)
        X_train, ids_train, run = load_embeddings(
            conn, model_name=model_name, view_name=view_name, run_id=run_id, result_ids=train_ids
        )
        if X_train.shape[0] == 0:
            raise SystemExit("No embeddings matched cleaned labels.")
        y_train = np.array([train_labels[int(rid)] for rid in ids_train])

        def _norm(mat: np.ndarray) -> np.ndarray:
            mat = mat.astype(np.float32, copy=False)
            if not bool(args.normalize):
                return mat
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return mat / norms

        X_train = _norm(X_train)
        print(f"embedding_run_id={run['embedding_run_id']}  dim={X_train.shape[1]}")
        print(f"train_samples={X_train.shape[0]}")
        unique, counts = np.unique(y_train, return_counts=True)
        print("train_class_counts=" + ", ".join(f"{lab}:{int(c)}" for lab, c in zip(unique, counts)))

        clf = LogisticRegression(max_iter=1000, C=float(args.C), class_weight="balanced")
        cv = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
        print(f"probe_cv5_accuracy={cv.mean():.4f} +/- {cv.std():.4f}")
        clf.fit(X_train, y_train)
        classes = list(clf.classes_)
        print()

        # 2. Predict the review_queue boxes.
        queue = _read_review_queue(conn, run_id=run_id, include_reject=bool(args.include_reject))
        if not queue:
            raise SystemExit("review_queue is empty for this run (with current filters).")
        queue_ids = sorted(queue)
        X_q, ids_q, _ = load_embeddings(
            conn, model_name=model_name, view_name=view_name, run_id=run_id, result_ids=queue_ids
        )
        missing = len(queue_ids) - len(ids_q)
        print(f"review_queue boxes={len(queue_ids)}  with_embedding={len(ids_q)}  missing_embedding={missing}")
        if X_q.shape[0] == 0:
            raise SystemExit("No embeddings matched review_queue boxes.")
        X_q = _norm(X_q)
        proba = clf.predict_proba(X_q)
        pred_idx = proba.argmax(axis=1)
        pred_label = np.array([classes[i] for i in pred_idx])
        max_prob = proba.max(axis=1)
        src_type = np.array([queue[int(rid)] for rid in ids_q])
        is_reject_src = src_type == "reject"
        print()

        # 3. Dry-run report across thresholds.
        print("threshold report (boxes that would be promoted):")
        print(f"  {'thresh':>6}  {'total':>7}  {'crack':>6}  {'mold':>6}  {'spall':>6}  {'from_reject':>11}")
        for thr in REPORT_THRESHOLDS:
            mask = max_prob >= thr
            sel_labels = pred_label[mask]
            dist = Counter(sel_labels.tolist())
            n_reject = int((mask & is_reject_src).sum())
            print(f"  {thr:>6.2f}  {int(mask.sum()):>7}  {dist.get('crack',0):>6}  "
                  f"{dist.get('mold',0):>6}  {dist.get('spall',0):>6}  {n_reject:>11}")
        print()

        if float(args.min_prob) <= 0:
            print("Report only (no --min-prob). Pick a threshold and re-run with --min-prob <v> --out-db <path>.")
            return 0

        # 4. Select promotions at the chosen threshold.
        thr = float(args.min_prob)
        sel_mask = max_prob >= thr
        sel_ids = [int(ids_q[i]) for i in range(len(ids_q)) if sel_mask[i]]
        sel_label_by_id = {int(ids_q[i]): str(pred_label[i]) for i in range(len(ids_q)) if sel_mask[i]}
        sel_prob_by_id = {int(ids_q[i]): float(max_prob[i]) for i in range(len(ids_q)) if sel_mask[i]}
        sel_src_by_id = {int(ids_q[i]): str(src_type[i]) for i in range(len(ids_q)) if sel_mask[i]}
        print(f"min_prob={thr}  selected={len(sel_ids)} boxes  "
              f"(from_reject={sum(1 for i in sel_ids if sel_src_by_id[i]=='reject')})")

        # 5. Fetch box geometry from crop_views (tight); only promote boxes with coords.
        coords: dict[int, tuple] = {}
        chunk = 900
        for start in range(0, len(sel_ids), chunk):
            part = sel_ids[start:start + chunk]
            ph = ",".join("?" for _ in part)
            for row in conn.execute(
                f"""SELECT result_id, image_rel_path, crop_path, x1, y1, x2, y2
                    FROM crop_views WHERE run_id=? AND view_name=? AND result_id IN ({ph})""",
                [run_id, view_name, *part],
            ).fetchall():
                if row["x1"] is None:
                    continue
                coords[int(row["result_id"])] = (
                    str(row["image_rel_path"] or ""), row["crop_path"],
                    float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]),
                )
        promotable = [rid for rid in sel_ids if rid in coords]
        print(f"promotable_with_coords={len(promotable)} (dropped {len(sel_ids)-len(promotable)} without tight box)")

        if not args.out_db:
            print()
            print("Dry-run (no --out-db). Re-run with --out-db <path> to write a standalone dataset SQLite.")
            return 0

        # 6. Build the standalone dataset SQLite: original cleaned + probe promotes.
        #    Source DB stays read-only; everything is written into out_db.
        orig_cleaned = conn.execute(
            """SELECT result_id, image_rel_path, crop_path, final_label, export_label,
                      decision_type, reliability_score, reason_codes_json, x1, y1, x2, y2, decision_policy_run_id
               FROM cleaned_labels WHERE run_id=?""",
            (run_id,),
        ).fetchall()
        run_row = conn.execute("SELECT * FROM resemi_runs WHERE run_id=?", (run_id,)).fetchone()
        run_cols = run_row.keys()
        run_vals = [run_row[c] for c in run_cols]
    finally:
        conn.close()

    policy_run_id = f"probe_{run_id}"
    out = connect_output(args.out_db)
    try:
        out.execute("DELETE FROM cleaned_labels WHERE run_id=?", (run_id,))
        out.execute("DELETE FROM resemi_runs WHERE run_id=?", (run_id,))
        ph = ",".join("?" for _ in run_cols)
        out.execute(f"INSERT OR REPLACE INTO resemi_runs ({', '.join(run_cols)}) VALUES ({ph})", run_vals)

        # Original cleaned rows (auto_accept / low_priority) copied verbatim.
        out.executemany(
            """INSERT OR REPLACE INTO cleaned_labels
               (run_id, result_id, image_rel_path, crop_path, final_label, export_label,
                decision_type, reliability_score, reason_codes_json, x1, y1, x2, y2, decision_policy_run_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(run_id, *[r[k] for k in (
                "result_id", "image_rel_path", "crop_path", "final_label", "export_label",
                "decision_type", "reliability_score", "reason_codes_json", "x1", "y1", "x2", "y2",
                "decision_policy_run_id")]) for r in orig_cleaned],
        )
        orig_ids = {int(r["result_id"]) for r in orig_cleaned}

        # Probe promotes (skip any that already exist as original cleaned).
        probe_rows = []
        for rid in promotable:
            if rid in orig_ids:
                continue
            label = sel_label_by_id[rid]
            prob = sel_prob_by_id[rid]
            image_rel_path, crop_path, x1, y1, x2, y2 = coords[rid]
            reasons = ["probe_relabel", f"probe_prob_{prob:.2f}", f"probe_src_{sel_src_by_id[rid]}"]
            probe_rows.append((
                run_id, rid, image_rel_path, crop_path, label, label,
                "probe_relabel", prob, json.dumps(reasons, ensure_ascii=False, sort_keys=True),
                x1, y1, x2, y2, policy_run_id,
            ))
        out.executemany(
            """INSERT OR REPLACE INTO cleaned_labels
               (run_id, result_id, image_rel_path, crop_path, final_label, export_label,
                decision_type, reliability_score, reason_codes_json, x1, y1, x2, y2, decision_policy_run_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            probe_rows,
        )
        total_cleaned = int(out.execute("SELECT COUNT(*) FROM cleaned_labels WHERE run_id=?", (run_id,)).fetchone()[0])
        out.execute("UPDATE resemi_runs SET cleaned_count=? WHERE run_id=?", (total_cleaned, run_id))
        out.commit()
    except Exception:
        out.rollback()
        raise
    finally:
        out.close()

    print(f"out_db={args.out_db}")
    print(f"wrote original_cleaned={len(orig_cleaned)} + probe_promotes={len(probe_rows)} = {total_cleaned}")
    dist = Counter()
    for r in orig_cleaned:
        dist[str(r["final_label"])] += 1
    for rid in promotable:
        if rid not in orig_ids:
            dist[sel_label_by_id[rid]] += 1
    print("cleaned_by_label=" + ", ".join(f"{k}:{v}" for k, v in sorted(dist.items())))
    print("source DB left unchanged (read-only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
