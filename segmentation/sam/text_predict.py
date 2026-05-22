"""Text-prompt SAM segmentation via GroundingDINO + SAM.

Usage:
    python -m segmentation.sam.text_predict --input /path/to/input.json

Input JSON:
    image_path        str   Path to input image
    text_prompt       str   Text query, e.g. "crack . peeling"
    sam_checkpoint    str   Path to SAM .pth
    gdino_checkpoint  str   Path to GroundingDINO checkpoint (or HF model id)
    model_type        str   "auto" | "vit_b" | "vit_l" | "vit_h"
    device            str   "auto" | "cpu" | "cuda" | "mps"
    box_threshold     float  default 0.30
    text_threshold    float  default 0.25
    output_dir        str

Output JSON (stdout):
    overlay_b64   str    PNG base64 for inline display
    overlay_path  str
    mask_path     str
    detections    int    number of detected regions
    model_type    str
    device        str
    error         str    only on failure
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        payload = json.load(f)

    image_path: str = payload["image_path"]
    text_prompt: str = payload.get("text_prompt") or ""
    sam_checkpoint: str = payload["sam_checkpoint"]
    gdino_checkpoint: str = payload.get("gdino_checkpoint") or ""
    model_type: str = payload.get("model_type") or "auto"
    device: str = payload.get("device") or "auto"
    box_threshold: float = float(payload.get("box_threshold") or 0.15)
    text_threshold: float = float(payload.get("text_threshold") or 0.15)
    output_dir: str = payload.get("output_dir") or "results_text_sam"

    try:
        import cv2
        import numpy as np
        from torch_runtime import select_device_str

        from object_detection.dino.client import get_dino_service
        from object_detection.dino.engine import default_gdino_checkpoint
        from segmentation.sam.backbones.segment_anything import SamPredictor
        from segmentation.sam.runtime.runtime import load_sam_model

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.path.isfile(sam_checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint}")
        if not text_prompt.strip():
            raise ValueError("text_prompt is required")

        checkpoint = gdino_checkpoint.strip() or str(default_gdino_checkpoint() or "").strip()
        if not checkpoint:
            raise RuntimeError("No GroundingDINO checkpoint available. Set gdino_checkpoint in the settings.")

        device_str = select_device_str(device)

        # ── Step 1: GroundingDINO detection ──────────────────────────────────
        print(json.dumps({"progress": "Running GroundingDINO detection..."}), flush=True)
        queries = [q.strip() for q in text_prompt.replace(",", ".").split(".") if q.strip()]
        params = {
            "gdino_checkpoint": checkpoint,
            "gdino_config_id": "auto",
            "text_queries": queries,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "max_dets": 80,
            "device": device,
            "output_dir": output_dir,
        }
        service = get_dino_service()
        try:
            result = service.call("predict", {"image_path": image_path, "params": params})
        finally:
            service.close()

        detections = list((result or {}).get("detections") or [])
        boxes = []
        for det in detections:
            b = det.get("box")
            if b and len(b) == 4:
                boxes.append([float(v) for v in b])

        if not boxes:
            raise ValueError(
                f"No detections for prompt: {text_prompt!r} "
                f"(box_threshold={box_threshold}, text_threshold={text_threshold}). "
                "Try a different prompt or lower the thresholds."
            )

        # ── Step 2: SAM on each box ───────────────────────────────────────────
        print(json.dumps({"progress": f"Found {len(boxes)} detection(s). Loading SAM..."}), flush=True)
        sam_model, used_type = load_sam_model(sam_checkpoint, model_type)
        sam_model.to(device=device_str)
        predictor = SamPredictor(sam_model)

        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)

        disp = bgr.copy()
        combined_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        overlay_color = np.array([120, 200, 0], dtype=np.uint8)

        print(json.dumps({"progress": f"Running SAM on {len(boxes)} box(es)..."}), flush=True)
        for box in boxes:
            np_box = np.array(box, dtype=np.float32)[None]
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np_box,
                multimask_output=True,
            )
            best = masks[int(np.argmax(scores))].astype(np.uint8)
            combined_mask = np.maximum(combined_mask, best)

        # Build overlay
        mask_layer = np.zeros_like(bgr)
        mask_layer[combined_mask > 0] = overlay_color
        disp = cv2.addWeighted(disp, 1.0, mask_layer, 0.45, 0)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(disp, contours, -1, (0, 220, 80), 2)

        # Draw bounding boxes + labels
        for det, box in zip(detections, boxes):
            x1, y1, x2, y2 = (int(v) for v in box)
            label = str(det.get("label") or "object")
            score = float(det.get("score") or 0.0)
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 200, 255), 2)
            text = f"{label} {score:.2f}"
            cv2.putText(disp, text, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(disp, text, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        overlay_path = os.path.join(output_dir, f"{base}_text_sam_overlay.png")
        mask_path = os.path.join(output_dir, f"{base}_text_sam_mask.png")
        cv2.imwrite(overlay_path, disp)
        cv2.imwrite(mask_path, combined_mask * 255)

        ok, buf = cv2.imencode(".png", disp)
        overlay_b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
        mask_ok, mask_buf = cv2.imencode(".png", combined_mask * 255)
        mask_b64 = base64.b64encode(mask_buf.tobytes()).decode("ascii") if mask_ok else None
        ys, xs = np.where(combined_mask > 0)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)] if xs.size else None

        print(json.dumps({
            "overlay_path": os.path.abspath(overlay_path),
            "mask_path": os.path.abspath(mask_path),
            "overlay_b64": overlay_b64,
            "mask_b64": mask_b64,
            "bbox": bbox,
            "detections": len(boxes),
            "model_type": used_type,
            "device": device_str,
        }), flush=True)

    except Exception as exc:
        import traceback
        print(json.dumps({"error": str(exc), "traceback": traceback.format_exc()}), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
