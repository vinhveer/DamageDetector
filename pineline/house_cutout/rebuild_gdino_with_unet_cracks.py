from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from segmentation.unet.engine import UnetParams, UnetRunner

from pineline.house_cutout.region_damage_overlays import (
    build_overlays,
    color_for,
    export_data,
    image_to_bgr_respecting_alpha,
    render_text_x3,
)


LAB = Path("/Users/nguyenquangvinh/Desktop/Lab")
RESULTS = LAB / "model_with_inference" / "semi_labeling_training" / "results"
UNET_V1 = LAB / "model_with_inference" / "crack_segmentation" / "unet_efficientnet_b4" / "model" / "best_model.pth"


def _row_box(row):
    if "full_box_x1" in row and row.get("full_box_x1"):
        return [float(row[f"full_box_{k}"]) for k in ("x1", "y1", "x2", "y2")]
    return [float(row[f"box_{k}"]) for k in ("x1", "y1", "x2", "y2")]


def _row_roi(row, box, width, height, pad=0):
    if all(row.get(k) for k in ("roi_x1", "roi_y1", "roi_x2", "roi_y2")):
        roi = [int(round(float(row[k]))) for k in ("roi_x1", "roi_y1", "roi_x2", "roi_y2")]
    else:
        roi = [int(np.floor(box[0])) - pad, int(np.floor(box[1])) - pad,
               int(np.ceil(box[2])) + pad, int(np.ceil(box[3])) + pad]
    roi[0] = max(0, min(width, roi[0])); roi[2] = max(0, min(width, roi[2]))
    roi[1] = max(0, min(height, roi[1])); roi[3] = max(0, min(height, roi[3]))
    return roi


def _load_rows(csv_path):
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mask_area(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0
    return int((mask > 127).sum())


def _copy_or_crop_mask(src_path, dst_path, roi, full_shape):
    mask = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
    h, w = full_shape[:2]
    x1, y1, x2, y2 = roi
    if mask.shape[:2] == (h, w):
        mask = mask[y1:y2, x1:x2]
    elif mask.shape[:2] != (y2 - y1, x2 - x1):
        mask = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(dst_path), mask)
    return True


def _unet_mask_for_roi(runner, image_path, params, roi, tmp_dir, out_mask_path):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    result = runner.run_rois(str(image_path), params, [tuple(roi)], log_fn=lambda s: print(s, flush=True))
    full_mask = cv2.imread(result["mask_path"], cv2.IMREAD_GRAYSCALE)
    if full_mask is None:
        raise RuntimeError(f"UNet did not write mask: {result.get('mask_path')}")
    x1, y1, x2, y2 = roi
    roi_mask = full_mask[y1:y2, x1:x2]
    cv2.imwrite(str(out_mask_path), roi_mask)
    return int((roi_mask > 127).sum())


def rebuild_case(*, name, image_path, src_dir, out_dir, csv_path=None, crop_box=None, extra_box=None, background=(0, 0, 0)):
    image_path = Path(image_path)
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    csv_path = Path(csv_path or src_dir / "detections_summary.csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "roi_images").mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "tmp_unet"

    full = image_to_bgr_respecting_alpha(image_path, background=background)
    h, w = full.shape[:2]
    cv2.imwrite(str(out_dir / "region_crop.png"), full if crop_box is None else full[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]])

    rows = _load_rows(csv_path)
    runner = UnetRunner()
    params = UnetParams(
        model_path=str(UNET_V1),
        output_dir=str(tmp_dir),
        threshold=0.5,
        apply_postprocessing=True,
        mode="tile",
        input_size=512,
        tile_overlap=256,
        tile_batch_size=4,
        device="auto",
    )

    seg_results = []
    for fallback_idx, row in enumerate(rows):
        group = str(row.get("group") or "damage").strip().lower()
        idx = int(row.get("idx") or row.get("rank") or fallback_idx)
        box = _row_box(row)
        roi = _row_roi(row, box, w, h, pad=0)
        if roi[2] <= roi[0] or roi[3] <= roi[1]:
            continue
        dst_mask = out_dir / "masks" / f"det_{idx:04d}_{group}.png"
        if group == "crack":
            area = _unet_mask_for_roi(runner, image_path, params, roi, tmp_dir, dst_mask)
            segmenter = "unet_v1"
        else:
            src_mask = row.get("mask_path") or ""
            if not src_mask:
                src_mask = src_dir / "masks" / f"det_{idx:04d}_{group}.png"
            copied = _copy_or_crop_mask(src_mask, dst_mask, roi, full.shape)
            area = _mask_area(dst_mask) if copied else int(float(row.get("mask_area_px") or 0))
            segmenter = row.get("segmenter") or "sam_vit_h"
        cv2.imwrite(str(out_dir / "roi_images" / f"det_{idx:04d}_{group}.png"), full[roi[1]:roi[3], roi[0]:roi[2]])
        seg_results.append({
            "idx": idx,
            "group": group,
            "label": row.get("label") or group,
            "score": float(row.get("score") or 0.0),
            "segmenter": segmenter,
            "mask_area_px": area,
            "box": box,
            "roi": roi,
        })

    seg_results.sort(key=lambda r: r["idx"])
    build_overlays(full, (0, 0), full, seg_results, out_dir, crop_box=crop_box, extra_box=extra_box)
    render_text_x3(out_dir, full, seg_results, crop_box, extra_box=extra_box)
    export_data(out_dir, image_path, (0, 0, w, h), {
        "source": name,
        "source_csv": str(csv_path),
        "crack_segmenter": "unet_v1",
        "non_crack_segmenter": "sam_vit_h_reused",
        "unet_model": str(UNET_V1),
    }, seg_results)
    print(f"[{name}] done -> {out_dir} detections={len(seg_results)}", flush=True)


def main():
    g8_column = [2680, 0, 3759, 3640]
    rebuild_case(
        name="g8_gdino_sam_vith_unet_crack",
        image_path=LAB / "data" / "HinhAnhThucTe" / "1.JPG",
        src_dir=RESULTS / "g8",
        out_dir=RESULTS / "g8" / "damage_segmentation_gdino_sam_vith_unet_crack",
        crop_box=None,
        extra_box=(*g8_column, "column"),
        background=(0, 0, 0),
    )
    rebuild_case(
        name="ntt_gdino_sam_vith_unet_crack",
        image_path=RESULTS / "nha_truyen_thong" / "damage_segmentation_clean_overlays" / "clean_house_bg.png",
        src_dir=RESULTS / "nha_truyen_thong" / "damage_segmentation_clean_overlays",
        out_dir=RESULTS / "nha_truyen_thong" / "damage_segmentation_gdino_sam_vith_unet_crack",
        crop_box=None,
        background=(255, 255, 255),
    )


if __name__ == "__main__":
    main()
