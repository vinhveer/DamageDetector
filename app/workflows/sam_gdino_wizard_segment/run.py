"""
SAM+GDino Wizard – Segmentation workflow
=========================================
Reads boxes from filtered.sqlite3 produced by wizard_detect.
Runs SAM (zero-shot or LoRA) per-image, saves PNG masks.
Prints RESULT_JSON:<json> at the end.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import uuid
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_APP_DIR = _HERE.parent
_REPO_ROOT = _APP_DIR.parent
for _p in [str(_REPO_ROOT), str(_APP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main(values: dict) -> None:
    db_path_raw: str = str(values.get("db_path", "") or "")
    sam_backend: str = str(values.get("sam_backend", "sam"))
    sam_model_type: str = str(values.get("sam_model_type", "vit_h"))
    sam_checkpoint: str = str(values.get("sam_checkpoint", "") or "")
    lora_checkpoint: str = str(values.get("lora_checkpoint", "") or "")
    device: str = str(values.get("device", "auto"))
    multimask: bool = bool(values.get("multimask", False))
    min_mask_area: int = int(values.get("min_mask_area", 0))
    expand_box_px: int = int(values.get("expand_box_px", 0))

    if not db_path_raw or not Path(db_path_raw).exists():
        print(f"[wizard-segment] ERROR: db_path not found: {db_path_raw!r}", flush=True)
        sys.exit(1)
    if not sam_checkpoint or not Path(sam_checkpoint).exists():
        print(f"[wizard-segment] ERROR: sam_checkpoint not found: {sam_checkpoint!r}", flush=True)
        sys.exit(1)

    db_path = Path(db_path_raw).resolve()
    runs_root = db_path.parent.parent  # .tmp/sam_gdino_wizard/<session>/
    output_dir = runs_root / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Read boxes grouped by absolute image path
    rows = conn.execute("""
        SELECT d.detection_id, d.label, d.score, d.x1, d.y1, d.x2, d.y2,
               i.path AS image_path, i.width, i.height
        FROM detections d JOIN images i ON i.image_id = d.image_id
        WHERE d.stage = 'final'
        ORDER BY i.path, d.score DESC
    """).fetchall()
    conn.close()

    by_image: dict[str, list] = {}
    for row in rows:
        img = str(row["image_path"])
        if img not in by_image:
            by_image[img] = []
        by_image[img].append(dict(row))

    print(f"[wizard-segment] images={len(by_image)}, total_boxes={len(rows)}", flush=True)

    # Load SAM model
    import cv2
    import numpy as np
    from torch_runtime import select_device_str
    from segmentation.sam.runtime.runtime import load_sam_model
    from segmentation.sam.backbones.segment_anything import SamPredictor

    device_str = select_device_str(device)
    print(f"[wizard-segment] loading SAM {sam_model_type} on {device_str}…", flush=True)
    sam_model, used_type = load_sam_model(sam_checkpoint, sam_model_type)
    sam_model.to(device=device_str)
    predictor = SamPredictor(sam_model)
    print(f"[wizard-segment] SAM ready (type={used_type})", flush=True)

    masks_by_image: dict[str, list] = {}

    for img_path_str, boxes in by_image.items():
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"[wizard-segment] WARNING: image not found: {img_path}", flush=True)
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[wizard-segment] WARNING: cannot read: {img_path}", flush=True)
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)
        h, w = bgr.shape[:2]

        img_stem = img_path.stem
        img_out_dir = output_dir / img_stem
        img_out_dir.mkdir(parents=True, exist_ok=True)

        img_masks = []
        for box in boxes:
            det_id = box["detection_id"]
            x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
            if expand_box_px > 0:
                x1 = max(0, x1 - expand_box_px); y1 = max(0, y1 - expand_box_px)
                x2 = min(w, x2 + expand_box_px); y2 = min(h, y2 + expand_box_px)

            np_box = np.array([x1, y1, x2, y2], dtype=np.float32)[None]
            try:
                masks, scores, _ = predictor.predict(
                    point_coords=None, point_labels=None,
                    box=np_box, multimask_output=multimask,
                )
            except Exception as exc:
                print(f"[wizard-segment] WARNING: SAM failed for det {det_id}: {exc}", flush=True)
                continue

            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx].astype(np.uint8)
            best_score = float(scores[best_idx])
            area = int(np.sum(best_mask))

            if min_mask_area > 0 and area < min_mask_area:
                continue

            # Save mask PNG
            mask_png = img_out_dir / f"{det_id}.png"
            cv2.imwrite(str(mask_png), best_mask * 255)

            # Base64 thumbnail (resize to ≤512 wide for inline display)
            thumb_mask = best_mask.copy()
            if w > 512:
                scale = 512 / w
                tw, th = int(w * scale), int(h * scale)
                thumb_mask = cv2.resize(thumb_mask, (tw, th), interpolation=cv2.INTER_NEAREST)

            # Build coloured mask overlay for thumbnail
            thumb_rgb = cv2.resize(rgb, (thumb_mask.shape[1], thumb_mask.shape[0]))
            overlay = thumb_rgb.copy()
            overlay[thumb_mask > 0] = (overlay[thumb_mask > 0] * 0.5 + np.array([60, 220, 80]) * 0.5).astype(np.uint8)
            ok, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else None

            img_masks.append({
                "detection_id": det_id,
                "label": box["label"],
                "score": best_score,
                "area": area,
                "mask_png_b64": b64,
                "box": {"x1": float(box["x1"]), "y1": float(box["y1"]), "x2": float(box["x2"]), "y2": float(box["y2"])},
            })

        masks_by_image[img_path_str] = img_masks
        print(f"[wizard-segment] {img_path.name}: {len(img_masks)} masks", flush=True)

    result = {
        "masks_by_image": masks_by_image,
        "output_dir": str(output_dir),
    }
    print("RESULT_JSON:" + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--values-json", required=True)
    args, _ = p.parse_known_args()
    with open(args.values_json, encoding="utf-8") as f:
        main(json.load(f))
