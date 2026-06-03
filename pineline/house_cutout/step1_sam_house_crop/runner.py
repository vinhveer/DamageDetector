from __future__ import annotations

import csv
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from object_detection.dino.client import get_dino_service
from object_detection.dino.engine import default_gdino_checkpoint

from segmentation.sam.runtime.engine import SamParams, SamRunner

from pineline.house_cutout.step1_sam_house_crop.gdino_house import (
    detect_house_and_negatives,
)
from pineline.house_cutout.step1_sam_house_crop.image_io import to_working_rgb
from pineline.house_cutout.step1_sam_house_crop.overlay import write_overlay
from pineline.house_cutout.step1_sam_house_crop.point_sampler import (
    point_in_box,
    sample_points_in_box,
)
from pineline.house_cutout.step1_sam_house_crop.prompts import (
    combined_text_queries,
)
from pineline.house_cutout.step1_sam_house_crop.sam_crop import (
    mask_bbox,
    predict_mask_with_points,
    union_masks,
    write_cutout_and_mask,
)
from pineline.house_cutout.step1_sam_house_crop.store import (
    ensure_schema,
    insert_crop,
    insert_detections,
    insert_image,
    insert_run_info,
)


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

_SUMMARY_FIELDS = [
    "image_rel_path", "image_id", "cutout_path", "mask_path",
    "house_box_count", "neg_box_count",
    "crop_x1", "crop_y1", "crop_x2", "crop_y2", "mask_area_px",
]


def _image_id_for(rel_path: str) -> str:
    return hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:12]


def _iter_images(input_path: Path) -> tuple[Path, list[Path]]:
    """Trả về (base_dir, danh sách ảnh).

    `input_path` có thể là một thư mục (quét đệ quy) hoặc một file ảnh đơn lẻ.
    base_dir dùng để tính rel path (và image_id).
    """
    if input_path.is_file():
        return input_path.parent, [input_path]
    images = sorted(
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
    return input_path, images


def resolve_gdino_checkpoint(raw: str | None) -> str:
    ckpt = str(raw or "").strip() or str(default_gdino_checkpoint() or "").strip()
    if not ckpt:
        raise RuntimeError(
            "Không có GroundingDINO checkpoint. Truyền --checkpoint hoặc chạy "
            "`python setup.py download_models --name grounding_dino_base`."
        )
    return ckpt


def run_step1(
    *,
    input_dir: Path,
    db_path: Path,
    work_dir: Path,
    cutouts_dir: Path,
    masks_dir: Path,
    overlay_dir: Path | None,
    summary_csv: Path | None,
    positive_queries: list[str],
    negative_queries: list[str],
    gdino_checkpoint: str | None = None,
    sam_checkpoint: Path | None = None,
    sam_model_type: str = "auto",
    device: str = "auto",
    work_max_side: int = 4096,
    box_threshold: float = 0.15,
    text_threshold: float = 0.15,
    max_dets: int = 50,
    tiled_threshold: int = 2048,
    score_floor: float = 0.20,
    points_per_box: int = 5,
    pad_px: int = 16,
    limit: int = 0,
    skip_existing: bool = False,
    write_overlays: bool = True,
    log: Callable[[str], None] | None = None,
    stop_checker: Callable[[], bool] | None = None,
) -> dict:
    log = log or (lambda s: print(s, flush=True))

    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"--input-dir không tồn tại: {input_dir}")
    if sam_checkpoint is None or not Path(sam_checkpoint).exists():
        raise FileNotFoundError(
            f"SAM checkpoint không tìm thấy: {sam_checkpoint}. "
            "Chạy `python setup.py download_models --name sam_vit_h` hoặc truyền --sam-checkpoint."
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    cutouts_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    if write_overlays and overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    base_dir, images = _iter_images(input_dir)
    if limit and limit > 0:
        images = images[:limit]
    if not images:
        raise RuntimeError(f"Không tìm thấy ảnh nào trong {input_dir}")
    log(f"Tìm thấy {len(images)} ảnh trong {input_dir}.")

    ckpt = resolve_gdino_checkpoint(gdino_checkpoint)
    text_queries = combined_text_queries(positive_queries, negative_queries)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    summary_rows: list[dict] = []
    cutout_count = 0

    conn = sqlite3.connect(str(db_path))
    service = get_dino_service()

    sam_params = SamParams(
        sam_checkpoint=str(sam_checkpoint),
        sam_model_type=sam_model_type,
        device=device,
    )
    sam_runner = SamRunner()
    predictor, device_str = sam_runner.ensure_model_loaded(sam_params, log_fn=log)

    try:
        ensure_schema(conn)
        existing_ids: set[str] = set()
        if skip_existing:
            existing_ids = {
                str(r[0]) for r in conn.execute(
                    "SELECT image_id FROM images WHERE run_id = ?", (run_id,)
                ).fetchall()
            }

        for idx, img_path in enumerate(images, start=1):
            if stop_checker is not None and stop_checker():
                log("Có yêu cầu dừng; thoát step1.")
                break

            rel_path = str(img_path.relative_to(base_dir))
            image_id = _image_id_for(rel_path)
            if image_id in existing_ids:
                log(f"[{idx}/{len(images)}] {rel_path}: đã có trong DB; bỏ qua.")
                continue

            # 1. tif/jpg/png -> working RGB (downscale nếu quá lớn)
            try:
                work = to_working_rgb(
                    img_path, work_dir / f"{image_id}.png", max_side=work_max_side,
                )
            except Exception as exc:
                log(f"[{idx}/{len(images)}] {rel_path}: lỗi đọc ảnh: {exc}")
                continue
            wh, ww = work.work_height, work.work_width

            # 2. GDINO: house (positive) + window/door (negative)
            try:
                house_boxes, neg_boxes = detect_house_and_negatives(
                    service, work.work_path,
                    width=ww, height=wh,
                    checkpoint=ckpt, text_queries=text_queries,
                    box_threshold=box_threshold, text_threshold=text_threshold,
                    max_dets=max_dets, device=device,
                    tiled_threshold=tiled_threshold, score_floor=score_floor,
                )
            except Exception as exc:
                log(f"[{idx}/{len(images)}] {rel_path}: lỗi GDINO: {exc}")
                continue

            # ghi detections (house + negative) trong không gian working
            det_rows: list[dict] = []
            tagged = [("house", d) for d in house_boxes] + [("negative", d) for d in neg_boxes]
            for d_idx, (role, det) in enumerate(tagged):
                x1, y1, x2, y2 = det["box"]
                det_rows.append({
                    "run_id": run_id, "image_id": image_id, "det_idx": d_idx,
                    "role": role, "label": det["label"],
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "score": det["score"],
                })
            if det_rows:
                insert_detections(conn, det_rows)

            if not house_boxes:
                insert_image(conn, {
                    "run_id": run_id, "image_id": image_id,
                    "image_path": str(img_path), "image_rel_path": rel_path,
                    "orig_width": work.orig_width, "orig_height": work.orig_height,
                    "work_width": ww, "work_height": wh, "scale": work.scale,
                    "house_box_count": 0, "neg_box_count": len(neg_boxes),
                    "has_cutout": 0,
                })
                conn.commit()
                log(f"[{idx}/{len(images)}] {rel_path}: không phát hiện nhà; bỏ qua.")
                continue

            # 3-4. SAM mask với điểm dương (house) + điểm âm (window/door)
            predictor.set_image(work.rgb)
            neg_points_all: list[tuple[float, float]] = []
            for nb in neg_boxes:
                neg_points_all.extend(
                    sample_points_in_box(
                        nb["box"], image_w=ww, image_h=wh,
                        points_per_box=points_per_box,
                    )
                )

            masks: list[np.ndarray] = []
            used_pos_points: list[tuple[float, float]] = []
            used_neg_points: list[tuple[float, float]] = []
            for hb in house_boxes:
                pos_points = sample_points_in_box(
                    hb["box"], image_w=ww, image_h=wh, points_per_box=points_per_box,
                )
                if not pos_points:
                    continue
                # chỉ giữ điểm âm nằm TRONG box house hiện tại (nhiễu cục bộ)
                local_neg = [p for p in neg_points_all if point_in_box(p, hb["box"])]
                try:
                    mask = predict_mask_with_points(
                        predictor, pos_points, local_neg, multimask_output=True,
                    )
                except Exception as exc:
                    log(f"  house box lỗi SAM: {exc}")
                    continue
                if mask.size == 0 or not mask.any():
                    continue
                masks.append(mask)
                used_pos_points.extend(pos_points)
                used_neg_points.extend(local_neg)

            has_cutout = 0
            crop_bbox = None
            union = None
            if masks:
                union = union_masks(masks)
                crop_bbox = mask_bbox(union, pad_px=pad_px)

            if crop_bbox is not None:
                cutout_path = cutouts_dir / f"{image_id}.png"
                mask_path = masks_dir / f"{image_id}.png"
                area = write_cutout_and_mask(
                    work.rgb, union, crop_bbox,
                    cutout_path=cutout_path, mask_path=mask_path,
                )
                cx1, cy1, cx2, cy2 = crop_bbox
                insert_crop(conn, {
                    "run_id": run_id, "image_id": image_id,
                    "cutout_path": str(cutout_path), "mask_path": str(mask_path),
                    "crop_x1": cx1, "crop_y1": cy1, "crop_x2": cx2, "crop_y2": cy2,
                    "mask_area_px": int(area), "source_box_count": len(masks),
                })
                has_cutout = 1
                cutout_count += 1
                summary_rows.append({
                    "image_rel_path": rel_path, "image_id": image_id,
                    "cutout_path": str(cutout_path), "mask_path": str(mask_path),
                    "house_box_count": len(house_boxes), "neg_box_count": len(neg_boxes),
                    "crop_x1": cx1, "crop_y1": cy1, "crop_x2": cx2, "crop_y2": cy2,
                    "mask_area_px": int(area),
                })
            else:
                log(f"[{idx}/{len(images)}] {rel_path}: SAM không tạo được mask hợp lệ.")

            # 6. overlay
            if write_overlays and overlay_dir is not None:
                try:
                    write_overlay(
                        work.rgb, union,
                        house_boxes=house_boxes, neg_boxes=neg_boxes,
                        positive_points=used_pos_points, negative_points=used_neg_points,
                        crop_bbox=crop_bbox,
                        out_path=overlay_dir / f"{image_id}.png",
                    )
                except Exception as exc:
                    log(f"  overlay lỗi: {exc}")

            insert_image(conn, {
                "run_id": run_id, "image_id": image_id,
                "image_path": str(img_path), "image_rel_path": rel_path,
                "orig_width": work.orig_width, "orig_height": work.orig_height,
                "work_width": ww, "work_height": wh, "scale": work.scale,
                "house_box_count": len(house_boxes), "neg_box_count": len(neg_boxes),
                "has_cutout": has_cutout,
            })
            conn.commit()
            log(
                f"[{idx}/{len(images)}] {rel_path}: house={len(house_boxes)} "
                f"neg={len(neg_boxes)} masks={len(masks)} cutout={'yes' if has_cutout else 'no'} "
                f"(work {ww}x{wh}, scale={work.scale:.3f})"
            )

        image_count = conn.execute(
            "SELECT COUNT(*) FROM images WHERE run_id = ?", (run_id,)
        ).fetchone()[0]

        insert_run_info(conn, {
            "run_id": run_id,
            "created_at_utc": run_id,
            "input_dir": str(input_dir),
            "gdino_checkpoint": ckpt,
            "sam_checkpoint": str(sam_checkpoint),
            "device": device_str,
            "work_max_side": int(work_max_side),
            "box_threshold": float(box_threshold),
            "text_threshold": float(text_threshold),
            "score_floor": float(score_floor),
            "points_per_box": int(points_per_box),
            "pad_px": int(pad_px),
            "image_count": int(image_count),
            "cutout_count": int(cutout_count),
        })
    finally:
        try:
            service.close()
        except Exception:
            pass
        conn.close()

    if summary_csv is not None:
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(summary_rows)

    log(
        f"step1 xong. run_id={run_id} images={image_count} cutouts={cutout_count} "
        f"db={db_path} cutouts_dir={cutouts_dir}"
    )
    return {
        "run_id": run_id,
        "db_path": str(db_path),
        "cutouts_dir": str(cutouts_dir),
        "masks_dir": str(masks_dir),
        "overlay_dir": str(overlay_dir) if (write_overlays and overlay_dir) else None,
        "summary_csv": str(summary_csv) if summary_csv is not None else None,
        "image_count": int(image_count),
        "cutout_count": int(cutout_count),
    }
