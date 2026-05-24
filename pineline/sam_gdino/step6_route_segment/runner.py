from __future__ import annotations

import csv
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from segmentation.sam.runtime.client import get_sam_service

from pineline.sam_gdino.step6_route_segment.overlay import (
    decode_mask_b64,
    save_mask_png,
    write_overlay,
)
from pineline.sam_gdino.step6_route_segment.sam_lora_predictor import PatchedSamLoraPredictor
from pineline.sam_gdino.step6_route_segment.store import (
    ensure_schema,
    insert_masks,
    insert_run_info,
    load_routed_detections,
)


CRACK_LABEL = "crack"


def _safe_name(parent_image_id: str, det_idx: int) -> str:
    return f"{parent_image_id}__det_{det_idx:04d}.png"


def _image_size(image_path: Path) -> tuple[int, int]:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    return int(h), int(w)


def _call_sam(
    service,
    *,
    image_path: str,
    params: dict,
    boxes: list[dict],
) -> dict:
    return service.call(
        "segment_boxes",
        {"image_path": image_path, "params": params, "boxes": boxes},
        log_fn=None,
    )


def run_step6(
    *,
    step3_db: Path,
    step4_db: Path,
    step5_db: Path,
    rgb_dir: Path,
    db_path: Path,
    masks_dir: Path,
    overlay_dir: Path | None,
    summary_csv: Path | None,
    sam_checkpoint: Path,
    sam_model_type: str = "auto",
    sam_lora_base: Path,
    sam_lora_delta: Path,
    sam_lora_middle_dim: int = 32,
    sam_lora_scaling_factor: float = 0.2,
    sam_lora_rank: int = 4,
    device: str = "auto",
    step3_run_id: str | None = "latest",
    step4_run_id: str | None = "latest",
    step5_run_id: str | None = "latest",
    write_overlays: bool = True,
    log: Callable[[str], None] | None = None,
    stop_checker: Callable[[], bool] | None = None,
) -> dict:
    log = log or (lambda s: print(s, flush=True))
    if not sam_checkpoint.exists():
        raise FileNotFoundError(f"SAM base checkpoint not found: {sam_checkpoint}")
    if not sam_lora_base.exists():
        raise FileNotFoundError(f"SAM-LoRA base checkpoint not found: {sam_lora_base}")
    if not sam_lora_delta.exists():
        raise FileNotFoundError(f"SAM-LoRA delta checkpoint not found: {sam_lora_delta}")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    if write_overlays and overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    s3_conn = sqlite3.connect(str(step3_db))
    s4_conn = sqlite3.connect(str(step4_db))
    s5_conn = sqlite3.connect(str(step5_db))
    try:
        s3_rid, s4_rid, s5_rid, detections = load_routed_detections(
            s3_conn, s4_conn, s5_conn,
            step3_run_id=step3_run_id,
            step4_run_id=step4_run_id,
            step5_run_id=step5_run_id,
        )
    finally:
        s3_conn.close()
        s4_conn.close()
        s5_conn.close()
    if not detections:
        raise RuntimeError("No detections survived steps 3-5 to segment.")

    by_parent: dict[str, list[dict]] = defaultdict(list)
    for det in detections:
        by_parent[det["parent_image_id"]].append(det)

    crack_count = sum(1 for d in detections if d["clip_label"] == CRACK_LABEL)
    non_crack_count = len(detections) - crack_count
    log(
        f"Routing {len(detections)} dets: crack={crack_count} -> SAM-LoRA, "
        f"other={non_crack_count} -> SAM zero-shot."
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mask_rows: list[dict] = []
    summary_rows: list[dict] = []

    sam_params_common = {
        "sam_checkpoint": str(sam_checkpoint),
        "sam_model_type": str(sam_model_type),
        "device": str(device),
        "output_dir": str(masks_dir / "_sam_tmp"),
        "task_group": "more_damage",
    }

    sam_service = get_sam_service()
    lora_predictor = None
    if crack_count > 0:
        log(f"Loading SAM-LoRA in-process (delta={sam_lora_delta.name})...")
        lora_predictor = PatchedSamLoraPredictor(
            sam_checkpoint=sam_lora_base,
            delta_checkpoint=sam_lora_delta,
            device=device,
            middle_dim=sam_lora_middle_dim,
            scaling_factor=sam_lora_scaling_factor,
            rank=sam_lora_rank,
            sam_model_type="auto",
        )
        log(
            f"SAM-LoRA ready (type={lora_predictor.model_type}, "
            f"image_size={lora_predictor.image_size}, decoder={lora_predictor.decoder_type})."
        )
    try:
        for parent_image_id, dets in by_parent.items():
            if stop_checker is not None and stop_checker():
                log("Stop requested; aborting step6.")
                break
            rgb_path = rgb_dir / f"{parent_image_id}.png"
            if not rgb_path.exists():
                log(f"  {parent_image_id}: missing RGB at {rgb_path}, skipping")
                continue
            try:
                H, W = _image_size(rgb_path)
                image_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            except Exception as exc:
                log(f"  {parent_image_id}: open error {exc}")
                continue

            crack_dets = [d for d in dets if d["clip_label"] == CRACK_LABEL]
            other_dets = [d for d in dets if d["clip_label"] != CRACK_LABEL]
            image_decoded: list[dict] = []

            def _record(det: dict, mask: np.ndarray, model_used: str):
                if mask is None or mask.size == 0 or mask.shape != (H, W):
                    log(f"    det {det['det_idx']}: empty/invalid mask, skip")
                    return
                mask_path = masks_dir / parent_image_id / _safe_name(parent_image_id, det["det_idx"])
                area = save_mask_png(mask, mask_path)
                x1, y1, x2, y2 = det["box"]
                mask_rows.append(
                    {
                        "run_id": run_id,
                        "parent_image_id": parent_image_id,
                        "det_idx": det["det_idx"],
                        "clip_label": det["clip_label"],
                        "model_used": model_used,
                        "mask_path": str(mask_path),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "mask_area_px": int(area),
                    }
                )
                summary_rows.append(
                    {
                        "parent_image_id": parent_image_id,
                        "det_idx": det["det_idx"],
                        "clip_label": det["clip_label"],
                        "clip_prob": det["clip_prob"],
                        "model_used": model_used,
                        "mask_path": str(mask_path),
                        "mask_area_px": int(area),
                    }
                )
                image_decoded.append(
                    {
                        "box": det["box"],
                        "clip_label": det["clip_label"],
                        "model_used": model_used,
                        "mask": mask.astype(np.uint8),
                    }
                )

            if other_dets:
                log(f"  {parent_image_id}: SAM zero-shot on {len(other_dets)} non-crack boxes")
                payload = _call_sam(
                    sam_service,
                    image_path=str(rgb_path),
                    params=sam_params_common,
                    boxes=[{"box": d["box"], "label": d["clip_label"], "score": d["clip_prob"]} for d in other_dets],
                )
                returned = list((payload or {}).get("detections") or [])
                for det, ret in zip(other_dets, returned):
                    mask = decode_mask_b64(ret.get("mask_b64") or "", expected_shape=(H, W))
                    _record(det, mask, "sam_zero_shot")

            if crack_dets and lora_predictor is not None:
                log(f"  {parent_image_id}: SAM-LoRA on {len(crack_dets)} crack boxes")
                results = lora_predictor.predict_boxes(
                    image_rgb,
                    [d["box"] for d in crack_dets],
                )
                for det, (mask, _score) in zip(crack_dets, results):
                    _record(det, mask, "sam_lora")

            if write_overlays and overlay_dir is not None and image_decoded:
                try:
                    write_overlay(
                        image_path=rgb_path,
                        detections=image_decoded,
                        out_path=overlay_dir / f"{parent_image_id}.png",
                    )
                except Exception as exc:
                    log(f"    overlay failed: {exc}")
    finally:
        try:
            sam_service.close()
        except Exception:
            pass

    conn = sqlite3.connect(str(db_path))
    try:
        ensure_schema(conn)
        if mask_rows:
            insert_masks(conn, mask_rows)
            conn.commit()
        insert_run_info(
            conn,
            {
                "run_id": run_id,
                "created_at_utc": run_id,
                "step3_db": str(step3_db),
                "step4_db": str(step4_db),
                "step5_db": str(step5_db),
                "step3_run_id": s3_rid,
                "step4_run_id": s4_rid,
                "step5_run_id": s5_rid,
                "sam_checkpoint": str(sam_checkpoint),
                "sam_model_type": sam_model_type,
                "sam_lora_base": str(sam_lora_base),
                "sam_lora_delta": str(sam_lora_delta),
                "device": str(device),
                "input_count": len(detections),
                "crack_count": int(crack_count),
                "non_crack_count": int(non_crack_count),
                "mask_count": len(mask_rows),
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
                    "parent_image_id", "det_idx", "clip_label", "clip_prob",
                    "model_used", "mask_path", "mask_area_px",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

    log(
        f"step6 done. run_id={run_id} input={len(detections)} masks={len(mask_rows)} "
        f"db={db_path}"
    )
    return {
        "run_id": run_id,
        "db_path": str(db_path),
        "input_count": len(detections),
        "mask_count": len(mask_rows),
        "crack_count": crack_count,
        "non_crack_count": non_crack_count,
    }
