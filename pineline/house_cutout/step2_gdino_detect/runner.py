from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pineline.common.detection import (
    MultiDetector,
    default_detection_config,
    resolve_gdino_checkpoint,
)
from pineline.common.segmentation import MultiSegmenter, default_segmentation_config, segmentation_model_metadata
from pineline.house_cutout.step2_gdino_detect.overlay import write_detector_overlays, write_overlay
from pineline.house_cutout.step2_gdino_detect.prompts import (
    PromptGroup,
    combined_queries,
    match_group_for_label,
)
from pineline.house_cutout.step2_gdino_detect.rgb_export import rgba_to_rgb_on_black
from pineline.house_cutout.step2_gdino_detect.store import (
    ensure_schema,
    insert_detections,
    insert_images,
    insert_prompt_groups,
    insert_run_info,
    insert_segmentations,
)


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

_SUMMARY_FIELDS = [
    "image_rel_path", "image_id", "detector_name", "det_idx", "group_name", "label",
    "query_label", "x1", "y1", "x2", "y2", "score",
]


def resolve_checkpoint(raw: str | None) -> str:
    return resolve_gdino_checkpoint(raw)


def _image_id_for(rel_path: str) -> str:
    return hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:12]


def _iter_input_images(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


def _detect_one(
    service,
    image_path: Path,
    *,
    width: int,
    height: int,
    checkpoint: str,
    queries: list[str],
    box_threshold: float,
    text_threshold: float,
    max_dets: int,
    device: str,
    tiled_threshold: int,
    tile_scales: list[str],
    recursive_max_depth: int,
    min_box_px: int,
) -> list[dict]:
    max_dim = max(int(width), int(height))
    params = {
        "gdino_checkpoint": checkpoint,
        "gdino_config_id": "auto",
        "text_queries": queries,
        "box_threshold": float(box_threshold),
        "text_threshold": float(text_threshold),
        "max_dets": int(max_dets),
        "device": device,
        "recursive_tile_scales": tile_scales,
    }
    if max_dim > tiled_threshold:
        result = service.call(
            "recursive_detect",
            {
                "image_path": str(image_path),
                "params": params,
                "target_labels": queries,
                "max_depth": int(recursive_max_depth),
                "min_box_px": int(min_box_px),
            },
        )
    else:
        result = service.call("predict", {"image_path": str(image_path), "params": params})
    return list(result.get("detections") or [])


def run_step1(
    *,
    input_dir: Path,
    db_path: Path,
    rgb_dir: Path,
    overlay_dir: Path | None,
    summary_csv: Path | None,
    prompt_groups: list[PromptGroup],
    checkpoint: str | None = None,
    detection_models: str | list[str] | None = None,
    yolo_model: str | None = None,
    stabledino_checkpoint: str | None = None,
    device: str = "auto",
    box_threshold: float = 0.10,
    text_threshold: float = 0.10,
    yolo_conf: float = 0.05,
    yolo_iou: float = 0.45,
    stabledino_conf: float = 0.05,
    max_dets: int = 150,
    tiled_threshold: int = 400,
    tile_scales: list[str] | None = None,
    recursive_max_depth: int = 2,
    min_box_px: int = 12,
    gdino_tile_batch_size: int = 0,
    gdino_service_workers: int = 0,
    gdino_service_queue_size: int = 0,
    gdino_service_batch_size: int = 0,
    gdino_service_device_ids: str | None = None,
    max_box_area_ratio: float = 0.50,
    limit: int = 0,
    source_run_id: str | None = None,
    skip_existing: bool = False,
    write_overlays: bool = True,
    run_segmentation: bool = True,
    segmentation_output_dir: Path | None = None,
    sam_segment_checkpoint: str | Path | None = None,
    sam_segment_model_type: str = "vit_b",
    unet_model: str | Path | None = None,
    sam_finetune_checkpoint: str | Path | None = None,
    sam_finetune_delta_type: str = "lora",
    sam_finetune_delta_checkpoint: str | Path | None = None,
    sam_finetune_model_type: str = "vit_b",
    segmentation_threshold: float = 0.5,
    segmentation_min_score: float = 0.2,
    log: Callable[[str], None] | None = None,
    stop_checker: Callable[[], bool] | None = None,
) -> dict:
    log = log or (lambda s: print(s, flush=True))
    if not prompt_groups:
        raise ValueError("Step 1 requires at least one prompt group.")
    tile_scales = tile_scales or ["small", "medium"]
    det_config = default_detection_config(
        models=detection_models,
        gdino_checkpoint=checkpoint,
        yolo_model=yolo_model,
        stabledino_checkpoint=stabledino_checkpoint,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        yolo_conf=yolo_conf,
        yolo_iou=yolo_iou,
        stabledino_conf=stabledino_conf,
        max_dets=max_dets,
        device=device,
        tiled_threshold=tiled_threshold,
        tile_scales=tile_scales,
        recursive_max_depth=recursive_max_depth,
        min_box_px=min_box_px,
        recursive_tile_batch_size=gdino_tile_batch_size,
        gdino_service_workers=gdino_service_workers,
        gdino_service_queue_size=gdino_service_queue_size,
        gdino_service_batch_size=gdino_service_batch_size,
        gdino_service_device_ids=gdino_service_device_ids,
        stabledino_output_dir=(db_path.parent / "stabledino_tmp"),
    )
    ckpt = det_config.gdino_checkpoint
    seg_config = default_segmentation_config(
        enabled=run_segmentation,
        output_dir=segmentation_output_dir or (db_path.parent / "segments"),
        sam_checkpoint=sam_segment_checkpoint,
        sam_model_type=sam_segment_model_type,
        unet_model=unet_model,
        sam_finetune_checkpoint=sam_finetune_checkpoint,
        sam_finetune_delta_type=sam_finetune_delta_type,
        sam_finetune_delta_checkpoint=sam_finetune_delta_checkpoint,
        sam_finetune_model_type=sam_finetune_model_type,
        threshold=segmentation_threshold,
        device=device,
    )

    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input-dir not found: {input_dir}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    if write_overlays and overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    images = _iter_input_images(input_dir)
    if limit and limit > 0:
        images = images[:limit]
    if not images:
        raise RuntimeError(f"No input images found under {input_dir}")
    log(f"Found {len(images)} images under {input_dir}.")

    queries = combined_queries(prompt_groups)
    run_id = str(source_run_id or "").strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    summary_rows: list[dict] = []
    image_rows: list[dict] = []
    detection_count = 0
    stored_image_count = 0
    stored_detection_count = 0
    stored_segmentation_count = 0

    conn = sqlite3.connect(str(db_path))
    detector = MultiDetector(det_config, log=log)
    segmenter = MultiSegmenter(seg_config, log=log)
    try:
        ensure_schema(conn)
        insert_prompt_groups(conn, run_id, prompt_groups)
        existing_image_ids: set[str] = set()
        if skip_existing:
            existing_image_ids = {
                str(row[0])
                for row in conn.execute(
                    "SELECT image_id FROM images WHERE run_id = ?",
                    (run_id,),
                ).fetchall()
            }

        # Phase A: export every RGBA cutout to an RGB-on-black crop. This is the
        # shared input that GDINO, YOLO and StableDINO all run on, so detector
        # box coordinates stay in the same frame.
        log("Phase A: exporting RGBA cutouts to RGB crops...")
        prepared: list[dict] = []
        for idx, img_path in enumerate(images, start=1):
            if stop_checker is not None and stop_checker():
                log("Stop requested; aborting RGB export.")
                break
            rel_path = str(img_path.relative_to(input_dir))
            image_id = _image_id_for(rel_path)
            if image_id in existing_image_ids:
                log(f"[{idx}/{len(images)}] {rel_path}: already in DB; skipped.")
                continue
            rgb_path = rgb_dir / f"{image_id}.png"
            # RGBA -> RGB, cropped to the valid (alpha>0) bbox so empty regions
            # are never fed to the model. Always regenerate to keep the crop
            # offsets in sync with this run.
            try:
                export = rgba_to_rgb_on_black(img_path, rgb_path)
            except Exception as exc:
                log(f"[{idx}/{len(images)}] {rel_path}: rgb export error: {exc}")
                continue
            if export.width <= 1 or export.height <= 1:
                log(f"[{idx}/{len(images)}] {rel_path}: empty cutout (no valid pixels); skipped.")
                continue
            prepared.append({
                "idx": idx, "img_path": img_path, "rel_path": rel_path,
                "image_id": image_id, "rgb_path": rgb_path, "export": export,
            })

        # Phase B: run StableDINO ONCE over the whole batch (single model load
        # instead of reloading Detectron2 per image).
        log(f"Phase B: StableDINO batch prepass for {len(prepared)} prepared images...")
        detector.prepare_stabledino(
            [item["rgb_path"] for item in prepared],
            names=[g.name for g in prompt_groups],
        )

        # Phase C: per-image GDINO + YOLO + StableDINO cache lookup + segmentation.
        log("Phase C: per-image detection + routed segmentation...")
        for item in prepared:
            if stop_checker is not None and stop_checker():
                log("Stop requested; aborting step1.")
                break
            idx = item["idx"]
            img_path = item["img_path"]
            rel_path = item["rel_path"]
            image_id = item["image_id"]
            rgb_path = item["rgb_path"]
            export = item["export"]
            w, h = export.width, export.height
            off_x, off_y = export.offset_x, export.offset_y

            # 2-3. detect through every configured detector.
            try:
                log(f"[{idx}/{len(images)}] {rel_path}: detection start ({w}x{h})")
                detections = detector.detect(
                    rgb_path,
                    width=w,
                    height=h,
                    queries=queries,
                    names=[g.name for g in prompt_groups],
                )
            except Exception as exc:
                log(f"[{idx}/{len(images)}] {rel_path}: detect error: {exc}")
                continue

            # 4. filter boxes
            rows: list[dict] = []
            tagged_detections: list[dict] = []
            segmentation_inputs: list[dict] = []
            image_area = max(1.0, float(w) * float(h))
            drop_too_large = 0
            for det_idx, det in enumerate(detections):
                box = det.get("box")
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = [float(v) for v in box]
                except (TypeError, ValueError):
                    continue
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                if (bw * bh) / image_area > float(max_box_area_ratio):
                    drop_too_large += 1
                    continue
                query_label = str(det.get("label") or "")
                detector_name = str(det.get("detector_name") or "gdino")
                score = float(det.get("score") or 0.0)
                match = match_group_for_label(query_label, prompt_groups)
                if match is None:
                    continue
                group_id, group_name = match
                # Boxes are in cropped RGB space (overlay uses this directly).
                # The DB/CSV store coordinates mapped back to the original
                # cutout so downstream consumers stay in the source frame.
                rows.append(
                    {
                        "run_id": run_id,
                        "image_id": image_id,
                        "det_idx": det_idx,
                        "detector_name": detector_name,
                        "group_id": group_id,
                        "group_name": group_name,
                        # one-word canonical label saved to DB/CSV
                        "label": group_name,
                        # raw GDINO phrase kept for debugging/traceability
                        "query_label": query_label,
                        "x1": x1 + off_x, "y1": y1 + off_y,
                        "x2": x2 + off_x, "y2": y2 + off_y,
                        "score": score,
                    }
                )
                tagged_detections.append(
                    {"box": [x1, y1, x2, y2], "score": score,
                     "label": f"{detector_name}:{group_name}", "detector_name": detector_name,
                     "group_name": group_name}
                )
                if score >= float(segmentation_min_score):
                    segmentation_inputs.append(
                        {
                            "detector_name": detector_name,
                            "det_idx": det_idx,
                            "label": group_name,
                            "score": score,
                            "box": [x1, y1, x2, y2],
                        }
                    )

            # 5. insert detections
            if rows:
                insert_detections(conn, rows)
                detection_count += len(rows)
                if run_segmentation:
                    try:
                        log(
                            f"  segmentation route start: {len(segmentation_inputs)}/{len(rows)} "
                            f"boxes score>={float(segmentation_min_score):.3f}"
                        )
                        seg_rows = segmenter.segment(rgb_path, segmentation_inputs)
                        for seg_row in seg_rows:
                            seg_row["run_id"] = run_id
                            seg_row["image_id"] = image_id
                        if seg_rows:
                            insert_segmentations(conn, seg_rows)
                        log(f"  segmentation route stored: {len(seg_rows)} rows")
                    except Exception as exc:
                        log(f"  segmentation failed: {exc}")
                # 6. overlay
                if write_overlays and overlay_dir is not None:
                    try:
                        write_overlay(
                            image_path=rgb_path,
                            detections=tagged_detections,
                            out_path=overlay_dir / f"{image_id}.png",
                        )
                        detector_paths = write_detector_overlays(
                            image_path=rgb_path,
                            detections=tagged_detections,
                            out_dir=overlay_dir / "by_model",
                            image_id=image_id,
                        )
                        if detector_paths:
                            log("  model overlays: " + ", ".join(f"{k}={v}" for k, v in sorted(detector_paths.items())))
                    except Exception as exc:
                        log(f"  overlay failed: {exc}")
                for row in rows:
                    summary_rows.append(
                        {
                            "image_rel_path": rel_path,
                            "image_id": image_id,
                            "detector_name": row["detector_name"],
                            "det_idx": row["det_idx"],
                            "group_name": row["group_name"],
                            "label": row["label"],
                            "query_label": row["query_label"],
                            "x1": row["x1"], "y1": row["y1"],
                            "x2": row["x2"], "y2": row["y2"],
                            "score": row["score"],
                        }
                    )

            image_rows.append(
                {
                    "run_id": run_id,
                    "image_id": image_id,
                    "image_path": str(img_path),
                    "image_rel_path": rel_path,
                    "width": int(w),
                    "height": int(h),
                    "orig_width": int(export.orig_width),
                    "orig_height": int(export.orig_height),
                    "offset_x": int(off_x),
                    "offset_y": int(off_y),
                    "det_count": len(rows),
                }
            )
            insert_images(conn, [image_rows[-1]])
            conn.commit()
            # 7. log
            log(
                f"[{idx}/{len(images)}] {rel_path}: raw={len(detections)} "
                f"matched={len(rows)} drop_large={drop_too_large} (image size {w}x{h})"
            )

        stored_image_count = conn.execute(
            "SELECT COUNT(*) FROM images WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0]
        stored_detection_count = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0]
        stored_segmentation_count = conn.execute(
            "SELECT COUNT(*) FROM segmentation_results WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0]

        insert_run_info(
            conn,
            {
                "run_id": run_id,
                "created_at_utc": created_at,
                "input_dir": str(input_dir),
                "checkpoint": ckpt,
                "detection_models": json.dumps(list(detector.active_models)),
                "yolo_model": det_config.yolo_model,
                "stabledino_checkpoint": det_config.stabledino_checkpoint,
                "device": device,
                "box_threshold": float(box_threshold),
                "text_threshold": float(text_threshold),
                "tile_scales": json.dumps(list(tile_scales)),
                "max_dets": int(max_dets),
                "image_count": int(stored_image_count),
                "detection_count": int(stored_detection_count),
                "segmentation_count": int(stored_segmentation_count),
                "segmentation_models": json.dumps(segmentation_model_metadata(seg_config), ensure_ascii=False),
            },
        )
    finally:
        try:
            detector.close()
            segmenter.close()
        except Exception:
            pass
        conn.close()

    if summary_csv is not None:
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        if skip_existing:
            conn = sqlite3.connect(str(db_path))
            try:
                summary_rows = [
                    dict(zip(_SUMMARY_FIELDS, row))
                    for row in conn.execute(
                        """
                        SELECT i.image_rel_path, d.image_id, d.detector_name, d.det_idx,
                               d.group_name, d.label, d.query_label,
                               d.x1, d.y1, d.x2, d.y2, d.score
                        FROM detections d
                        JOIN images i
                          ON i.run_id = d.run_id AND i.image_id = d.image_id
                        WHERE d.run_id = ?
                        ORDER BY i.image_rel_path, d.detector_name, d.det_idx
                        """,
                        (run_id,),
                    ).fetchall()
                ]
            finally:
                conn.close()
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(summary_rows)

    log(
        f"step1 done. run_id={run_id} images={stored_image_count} "
        f"dets={stored_detection_count} segments={stored_segmentation_count} db={db_path}"
    )
    return {
        "run_id": run_id,
        "db_path": str(db_path),
        "rgb_dir": str(rgb_dir),
        "overlay_dir": str(overlay_dir) if (write_overlays and overlay_dir) else None,
        "summary_csv": str(summary_csv) if summary_csv is not None else None,
        "image_count": stored_image_count,
        "detection_count": stored_detection_count,
        "segmentation_count": stored_segmentation_count,
    }
