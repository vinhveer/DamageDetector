from __future__ import annotations

import csv
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from pineline.sam_gdino.step5_embedding.dedup import dedup_within_image
from pineline.sam_gdino.step5_embedding.embedder import DinoV2Embedder
from pineline.sam_gdino.step5_embedding.store import (
    ensure_schema,
    insert_decisions,
    insert_run_info,
    load_step3_with_step4,
)


def _crop_box(image: Image.Image, box: list[float]) -> Image.Image:
    x1, y1, x2, y2 = box
    w, h = image.size
    ix1 = max(0, int(round(x1)))
    iy1 = max(0, int(round(y1)))
    ix2 = min(int(w), int(round(x2)))
    iy2 = min(int(h), int(round(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        return Image.new("RGB", (8, 8), color=(0, 0, 0))
    return image.crop((ix1, iy1, ix2, iy2))


def run_step5(
    *,
    step3_db: Path,
    step4_db: Path,
    rgb_dir: Path,
    db_path: Path,
    summary_csv: Path | None,
    step3_run_id: str | None = "latest",
    step4_run_id: str | None = "latest",
    model_name: str = "facebook/dinov2-small",
    device: str = "auto",
    batch_size: int = 16,
    iou_threshold: float = 0.50,
    cosine_threshold: float = 0.85,
    log: Callable[[str], None] | None = None,
    stop_checker: Callable[[], bool] | None = None,
) -> dict:
    log = log or (lambda s: print(s, flush=True))
    db_path.parent.mkdir(parents=True, exist_ok=True)

    s3_conn = sqlite3.connect(str(step3_db))
    s4_conn = sqlite3.connect(str(step4_db))
    try:
        s3_rid, s4_rid, detections = load_step3_with_step4(
            s3_conn, s4_conn,
            step3_run_id=step3_run_id, step4_run_id=step4_run_id,
        )
    finally:
        s3_conn.close()
        s4_conn.close()
    if not detections:
        raise RuntimeError(
            f"No labelled detections found "
            f"(step3={step3_db}, step4={step4_db})"
        )
    log(
        f"Loaded {len(detections)} detections "
        f"(step3 run {s3_rid}, step4 run {s4_rid})."
    )

    by_parent: dict[str, list[dict]] = defaultdict(list)
    for det in detections:
        by_parent[det["parent_image_id"]].append(det)

    log(f"Loading DINOv2 embedder: {model_name}")
    embedder = DinoV2Embedder(model_name=model_name, device=device)
    log(f"DINOv2 ready (device={embedder.device}).")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    all_decisions: list[dict] = []
    summary_rows: list[dict] = []

    for parent_image_id, dets in by_parent.items():
        if stop_checker is not None and stop_checker():
            log("Stop requested; aborting step5.")
            break
        rgb_path = rgb_dir / f"{parent_image_id}.png"
        if not rgb_path.exists():
            log(f"  {parent_image_id}: missing RGB at {rgb_path}, skipping")
            continue
        try:
            full = Image.open(rgb_path).convert("RGB")
        except Exception as exc:
            log(f"  {parent_image_id}: open error {exc}")
            continue
        crops = [_crop_box(full, d["box"]) for d in dets]
        embeddings = embedder.embed(crops, batch_size=batch_size)
        decisions = dedup_within_image(
            dets, embeddings,
            iou_threshold=iou_threshold,
            cosine_threshold=cosine_threshold,
        )
        kept = sum(d["kept"] for d in decisions)
        merged = len(decisions) - kept
        log(f"  {parent_image_id}: in={len(dets)} kept={kept} merged={merged}")
        for det, dec in zip(dets, decisions):
            row = {
                "run_id": run_id,
                **dec,
            }
            all_decisions.append(row)
            summary_rows.append(
                {
                    "parent_image_id": dec["parent_image_id"],
                    "det_idx": dec["det_idx"],
                    "kept": dec["kept"],
                    "merged_into_det_idx": dec["merged_into_det_idx"],
                    "gdino_group": det["gdino_group"],
                    "clip_label": det["clip_label"],
                    "clip_prob": det["clip_prob"],
                    "gdino_score": det["gdino_score"],
                    "reason": dec["reason"],
                }
            )

    conn = sqlite3.connect(str(db_path))
    try:
        ensure_schema(conn)
        if all_decisions:
            insert_decisions(conn, all_decisions)
            conn.commit()
        kept_count = sum(1 for d in all_decisions if d["kept"] == 1)
        insert_run_info(
            conn,
            {
                "run_id": run_id,
                "created_at_utc": run_id,
                "step3_db": str(step3_db),
                "step4_db": str(step4_db),
                "step3_run_id": s3_rid,
                "step4_run_id": s4_rid,
                "model_name": model_name,
                "device": embedder.device,
                "iou_threshold": float(iou_threshold),
                "cosine_threshold": float(cosine_threshold),
                "input_count": len(all_decisions),
                "kept_count": int(kept_count),
                "merged_count": int(len(all_decisions) - kept_count),
            },
        )
    finally:
        conn.close()

    if summary_csv is not None and summary_rows:
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "parent_image_id", "det_idx", "kept", "merged_into_det_idx",
                    "gdino_group", "clip_label", "clip_prob", "gdino_score", "reason",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

    log(
        f"step5 done. run_id={run_id} in={len(all_decisions)} "
        f"kept={sum(1 for d in all_decisions if d['kept']==1)} db={db_path}"
    )
    return {
        "run_id": run_id,
        "db_path": str(db_path),
        "input_count": len(all_decisions),
        "kept_count": int(sum(1 for d in all_decisions if d["kept"] == 1)),
    }
