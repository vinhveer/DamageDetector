from __future__ import annotations

import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from segmentation.sam.runtime.engine import SamParams, SamRunner

from pineline.sam_gdino.step2_sam_bridge_crop.point_sampler import sample_points_in_box
from pineline.sam_gdino.step2_sam_bridge_crop.sam_crop import (
    load_image_rgb,
    mask_bbox,
    predict_mask_for_box,
    union_masks,
    write_crop_and_mask,
    write_overlay,
)
from pineline.sam_gdino.step2_sam_bridge_crop.store import (
    ensure_schema,
    insert_crop,
    insert_run_info,
    load_kept_images,
)


def _safe_stem(rel_path: str) -> str:
    return Path(rel_path).stem.replace("/", "__").replace("\\", "__")


def run_step2(
    *,
    step1_db: Path,
    db_path: Path,
    crops_dir: Path,
    masks_dir: Path,
    overlay_dir: Path | None,
    summary_csv: Path | None,
    sam_checkpoint: Path,
    sam_model_type: str = "auto",
    device: str = "auto",
    points_per_box: int = 5,
    pad_px: int = 16,
    write_overlays: bool = True,
    source_run_id: str | None = None,
    log: Callable[[str], None] | None = None,
    stop_checker: Callable[[], bool] | None = None,
) -> dict:
    log = log or (lambda s: print(s, flush=True))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    if write_overlays and overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    if not sam_checkpoint.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found at {sam_checkpoint}. "
            "Run `python setup.py download_models --name sam_vit_h` or pass --sam-checkpoint."
        )

    src_conn = sqlite3.connect(str(step1_db))
    images = load_kept_images(src_conn, source_run_id=source_run_id)
    src_conn.close()
    if not images:
        raise RuntimeError(
            f"No kept images found in {step1_db} for source_run_id={source_run_id}"
        )
    log(f"Loaded {len(images)} kept images from step1.")

    params = SamParams(
        sam_checkpoint=str(sam_checkpoint),
        sam_model_type=sam_model_type,
        device=device,
    )
    runner = SamRunner()
    predictor, device_str = runner.ensure_model_loaded(params, log_fn=log)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_rows: list[dict] = []
    crop_count = 0

    conn = sqlite3.connect(str(db_path))
    try:
        ensure_schema(conn)
        for idx, img in enumerate(images, start=1):
            if stop_checker is not None and stop_checker():
                log("Stop requested; aborting step2.")
                break
            image_id = img["image_id"]
            image_path = Path(img["image_path"])
            rel = img["image_rel_path"]
            try:
                image_rgb = load_image_rgb(image_path)
            except Exception as exc:
                log(f"[{idx}/{len(images)}] error reading {rel}: {exc}")
                continue
            h, w = image_rgb.shape[:2]

            predictor.set_image(image_rgb)

            points_by_box: list[list[tuple[float, float]]] = []
            masks: list[np.ndarray] = []
            for b in img["boxes"]:
                pts = sample_points_in_box(
                    (b["x1"], b["y1"], b["x2"], b["y2"]),
                    image_w=w, image_h=h,
                    points_per_box=points_per_box,
                )
                if not pts:
                    continue
                try:
                    mask = predict_mask_for_box(predictor, pts, multimask_output=True)
                except Exception as exc:
                    log(f"  box {b['box_idx']}: SAM error: {exc}")
                    continue
                if mask.size == 0 or not mask.any():
                    continue
                points_by_box.append(pts)
                masks.append(mask)

            if not masks:
                log(f"[{idx}/{len(images)}] {rel}: no valid mask from any box")
                continue

            union = union_masks(masks)
            bbox = mask_bbox(union, pad_px=pad_px)
            if bbox is None:
                log(f"[{idx}/{len(images)}] {rel}: empty union mask")
                continue

            stem = _safe_stem(rel)
            crop_path = crops_dir / f"{image_id}__{stem}.png"
            mask_path = masks_dir / f"{image_id}__{stem}.png"
            area = write_crop_and_mask(
                image_rgb, union, bbox,
                crop_path=crop_path, mask_path=mask_path,
            )

            if write_overlays and overlay_dir is not None:
                overlay_path = overlay_dir / f"{image_id}__{stem}.png"
                try:
                    write_overlay(
                        image_rgb, union, points_by_box, bbox,
                        out_path=overlay_path,
                    )
                except Exception as exc:
                    log(f"  overlay failed: {exc}")

            x1, y1, x2, y2 = bbox
            insert_crop(
                conn,
                {
                    "run_id": run_id,
                    "parent_image_id": image_id,
                    "parent_image_path": str(image_path),
                    "crop_path": str(crop_path),
                    "mask_path": str(mask_path),
                    "crop_x1": x1, "crop_y1": y1, "crop_x2": x2, "crop_y2": y2,
                    "mask_area_px": area,
                    "source_box_count": len(masks),
                },
            )
            conn.commit()
            summary_rows.append(
                {
                    "parent_image_id": image_id,
                    "image_rel_path": rel,
                    "crop_path": str(crop_path),
                    "mask_path": str(mask_path),
                    "crop_x1": x1, "crop_y1": y1, "crop_x2": x2, "crop_y2": y2,
                    "mask_area_px": area,
                    "source_box_count": len(masks),
                }
            )
            crop_count += 1
            log(
                f"[{idx}/{len(images)}] {rel}: boxes={len(img['boxes'])} "
                f"masks={len(masks)} bbox=({x1},{y1},{x2},{y2}) area={area}"
            )

        insert_run_info(
            conn,
            {
                "run_id": run_id,
                "created_at_utc": run_id,
                "source_db": str(step1_db),
                "source_run_id": images[0]["source_run_id"],
                "sam_checkpoint": str(sam_checkpoint),
                "sam_model_type": sam_model_type,
                "device": device_str,
                "points_per_box": int(points_per_box),
                "pad_px": int(pad_px),
                "image_count": len(images),
                "crop_count": crop_count,
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
                    "parent_image_id", "image_rel_path",
                    "crop_path", "mask_path",
                    "crop_x1", "crop_y1", "crop_x2", "crop_y2",
                    "mask_area_px", "source_box_count",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

    log(
        f"step2 done. run_id={run_id} images={len(images)} crops={crop_count} "
        f"db={db_path}"
    )
    return {
        "run_id": run_id,
        "db_path": str(db_path),
        "crops_dir": str(crops_dir),
        "masks_dir": str(masks_dir),
        "overlay_dir": str(overlay_dir) if (write_overlays and overlay_dir) else None,
        "image_count": len(images),
        "crop_count": crop_count,
    }
