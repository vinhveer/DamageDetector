from __future__ import annotations

import csv
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import sqlite3 as _sqlite3
from PIL import Image

from pineline.sam_gdino.step4_openclip_semantic.clip_model import OpenClipClassifier
from pineline.sam_gdino.step4_openclip_semantic.prompts import DEFAULT_LABEL_PROMPTS
from pineline.sam_gdino.step4_openclip_semantic.store import (
    ensure_schema,
    insert_labels,
    insert_run_info,
    load_step3_detections,
)


def _resolve_image_path_for(parent_image_id: str, *, rgb_dir: Path) -> Path:
    return rgb_dir / f"{parent_image_id}.png"


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


def run_step4(
    *,
    step3_db: Path,
    rgb_dir: Path,
    db_path: Path,
    crops_dir: Path | None,
    summary_csv: Path | None,
    source_run_id: str | None = "latest",
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    device: str = "auto",
    batch_size: int = 16,
    save_crops: bool = False,
    min_prob: float = 0.45,
    log: Callable[[str], None] | None = None,
    stop_checker: Callable[[], bool] | None = None,
) -> dict:
    log = log or (lambda s: print(s, flush=True))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if save_crops and crops_dir is not None:
        crops_dir.mkdir(parents=True, exist_ok=True)

    src_conn = sqlite3.connect(str(step3_db))
    step3_run_id, detections = load_step3_detections(src_conn, source_run_id=source_run_id)
    src_conn.close()
    if not detections:
        raise RuntimeError(f"No damage detections found in {step3_db} run={source_run_id}")
    log(f"Loaded {len(detections)} detections from step3 run {step3_run_id}.")

    by_parent: dict[str, list[dict]] = defaultdict(list)
    for det in detections:
        by_parent[det["parent_image_id"]].append(det)

    log("Loading OpenCLIP model...")
    classifier = OpenClipClassifier(
        model_name=model_name, pretrained=pretrained, device=device,
        prompt_groups=DEFAULT_LABEL_PROMPTS,
    )
    log(f"CLIP ready (model={model_name}/{pretrained}, device={classifier.device}).")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label_rows: list[dict] = []
    summary_rows: list[dict] = []
    dropped_low_prob = 0

    conn = sqlite3.connect(str(db_path))
    try:
        ensure_schema(conn)
        for parent_image_id, dets in by_parent.items():
            if stop_checker is not None and stop_checker():
                log("Stop requested; aborting step4.")
                break
            rgb_path = _resolve_image_path_for(parent_image_id, rgb_dir=rgb_dir)
            if not rgb_path.exists():
                log(f"  {parent_image_id}: missing RGB at {rgb_path}, skipping")
                continue
            try:
                full = Image.open(rgb_path).convert("RGB")
            except Exception as exc:
                log(f"  {parent_image_id}: open error {exc}")
                continue

            crops = [_crop_box(full, d["box"]) for d in dets]
            if save_crops and crops_dir is not None:
                sub = crops_dir / parent_image_id
                sub.mkdir(parents=True, exist_ok=True)
                for d, im in zip(dets, crops):
                    im.save(sub / f"det_{d['det_idx']:04d}.png")

            # Batch CLIP classification.
            preds: list[dict] = []
            for start in range(0, len(crops), int(max(1, batch_size))):
                chunk = crops[start: start + int(max(1, batch_size))]
                preds.extend(classifier.classify_batch(chunk))

            for det, pred in zip(dets, preds):
                if float(pred["predicted_probability"]) < float(min_prob):
                    dropped_low_prob += 1
                    continue
                row = {
                    "run_id": run_id,
                    "parent_image_id": det["parent_image_id"],
                    "det_idx": det["det_idx"],
                    "predicted_label": pred["predicted_label"],
                    "predicted_probability": pred["predicted_probability"],
                    "class_scores_json": json.dumps(pred["class_scores"]),
                }
                label_rows.append(row)
                summary_rows.append(
                    {
                        "parent_image_id": det["parent_image_id"],
                        "det_idx": det["det_idx"],
                        "gdino_group": det["group_name"],
                        "gdino_label": det["label"],
                        "gdino_score": det["score"],
                        "clip_label": pred["predicted_label"],
                        "clip_prob": pred["predicted_probability"],
                    }
                )
            log(
                f"  {parent_image_id}: classified {len(dets)} boxes "
                f"(kept {len([d for d, p in zip(dets, preds) if p['predicted_probability'] >= min_prob])})"
            )

        if label_rows:
            insert_labels(conn, label_rows)
            conn.commit()
        insert_run_info(
            conn,
            {
                "run_id": run_id,
                "created_at_utc": run_id,
                "step3_db": str(step3_db),
                "step3_run_id": step3_run_id,
                "model_name": model_name,
                "pretrained": pretrained,
                "device": classifier.device,
                "batch_size": int(batch_size),
                "detection_count": len(label_rows),
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
                    "parent_image_id", "det_idx",
                    "gdino_group", "gdino_label", "gdino_score",
                    "clip_label", "clip_prob",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

    log(
        f"step4 done. run_id={run_id} labels={len(label_rows)} "
        f"dropped_low_prob<{min_prob}={dropped_low_prob} db={db_path}"
    )
    return {
        "run_id": run_id,
        "db_path": str(db_path),
        "detection_count": len(label_rows),
        "dropped_low_prob": dropped_low_prob,
    }
