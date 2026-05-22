"""Point-based SAM segmentation.

Usage:
    python -m segmentation.sam.point_predict --input /path/to/input.json

Input JSON schema:
    image_path    str       Path to input image
    points        [[x,y]]   Coordinate list (image space)
    labels        [0|1]     1 = positive, 0 = negative
    box           [x1,y1,x2,y2]  Optional bounding box
    sam_checkpoint str      Path to SAM .pth weight file
    model_type    str       "auto" | "vit_b" | "vit_l" | "vit_h"
    device        str       "auto" | "cpu" | "cuda" | "mps"
    output_dir    str       Directory to write result images

Output JSON (stdout):
    overlay_path  str
    mask_path     str
    overlay_b64   str       PNG base64 for inline display
    score         float
    mask_area     int
    model_type    str
    device        str
    error         str       Only present on failure
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM point-based segmentation")
    parser.add_argument("--input", required=True, help="Path to JSON input file")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        payload = json.load(f)

    image_path: str = payload["image_path"]
    points: list[list[float]] = payload.get("points") or []
    labels: list[int] = payload.get("labels") or []
    box: list[float] | None = payload.get("box")
    sam_checkpoint: str = payload["sam_checkpoint"]
    model_type: str = payload.get("model_type") or "auto"
    device: str = payload.get("device") or "auto"
    output_dir: str = payload.get("output_dir") or "results_point_sam"

    try:
        import cv2
        import numpy as np
        from torch_runtime import select_device_str

        from segmentation.sam.runtime.runtime import load_sam_model
        from segmentation.sam.backbones.segment_anything import SamPredictor

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.path.isfile(sam_checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint}")
        if not points and not box:
            raise ValueError("At least one point or a bounding box is required")
        if points and len(points) != len(labels):
            raise ValueError("points and labels must have the same length")

        device_str = select_device_str(device)

        print(json.dumps({"progress": f"Loading SAM model ({model_type}) on {device_str}..."}), flush=True)
        sam_model, used_type = load_sam_model(sam_checkpoint, model_type)
        sam_model.to(device=device_str)
        predictor = SamPredictor(sam_model)
        print(json.dumps({"progress": f"SAM ready (type={used_type}). Running prediction..."}), flush=True)

        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)

        np_points = np.array(points, dtype=np.float32) if points else None
        np_labels = np.array(labels, dtype=np.int32) if labels else None
        np_box = np.array(box, dtype=np.float32)[None] if box else None  # shape (1, 4)

        masks, scores, _logits = predictor.predict(
            point_coords=np_points,
            point_labels=np_labels,
            box=np_box,
            multimask_output=True,
        )

        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx].astype(np.uint8)
        best_score = float(scores[best_idx])

        # Build overlay: semi-transparent teal fill + contour
        disp = bgr.copy()
        overlay_color = np.array([120, 200, 0], dtype=np.uint8)  # BGR teal-green
        mask_layer = np.zeros_like(bgr)
        mask_layer[best_mask > 0] = overlay_color
        disp = cv2.addWeighted(disp, 1.0, mask_layer, 0.45, 0)

        contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(disp, contours, -1, (0, 220, 80), 2)

        # Draw points
        for (px, py), lbl in zip(points, labels):
            dot_bgr = (0, 220, 80) if lbl == 1 else (60, 60, 240)  # green positive, red negative
            cv2.circle(disp, (int(px), int(py)), 7, dot_bgr, -1)
            cv2.circle(disp, (int(px), int(py)), 7, (255, 255, 255), 2)

        # Draw box if provided
        if box:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 200, 255), 2)

        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        overlay_path = os.path.join(output_dir, f"{base}_point_sam_overlay.png")
        mask_path = os.path.join(output_dir, f"{base}_point_sam_mask.png")
        cv2.imwrite(overlay_path, disp)
        cv2.imwrite(mask_path, best_mask * 255)

        ok, buf = cv2.imencode(".png", disp)
        overlay_b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
        mask_ok, mask_buf = cv2.imencode(".png", best_mask * 255)
        mask_b64 = base64.b64encode(mask_buf.tobytes()).decode("ascii") if mask_ok else None
        ys, xs = np.where(best_mask > 0)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)] if xs.size else None

        print(json.dumps({
            "overlay_path": os.path.abspath(overlay_path),
            "mask_path": os.path.abspath(mask_path),
            "overlay_b64": overlay_b64,
            "mask_b64": mask_b64,
            "bbox": bbox,
            "score": best_score,
            "mask_area": int(np.sum(best_mask)),
            "model_type": used_type,
            "device": device_str,
        }), flush=True)

    except Exception as exc:
        import traceback
        print(json.dumps({"error": str(exc), "traceback": traceback.format_exc()}), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
