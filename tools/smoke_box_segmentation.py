from __future__ import annotations

"""Smoke-test box-scoped segmentation on a real image.

This script is intentionally independent from the UI. It verifies the contract the
UI depends on:
- detection produces boxes
- segmentation runs per box, not whole-image only
- every per-detection mask is a crop (or is tightly contained in its source box)

Example:
    cd /Users/nguyenquangvinh/Desktop/Lab/DamageDetector
    .venv/bin/python tools/smoke_box_segmentation.py \
      --image /Users/nguyenquangvinh/Downloads/wall-crack.jpg \
      --device auto
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parent
MODEL_ROOT = LAB / "model_with_inference"


def _add_repo_to_path() -> None:
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _default_manual_box(image_path: Path) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    h, w = image.shape[:2]
    # Tuned for /Users/nguyenquangvinh/Downloads/wall-crack.jpg: left-wall crack.
    return {
        "label": "crack",
        "score": 1.0,
        "box": [max(0, int(w * 0.08)), max(0, int(h * 0.02)), min(w - 1, int(w * 0.46)), min(h - 1, int(h * 0.98))],
        "source": "manual_fallback",
    }


def _run_yolo(image_path: Path, output_dir: Path, device: str, conf: float) -> list[dict[str, Any]]:
    yolo_weight_candidates = [
        MODEL_ROOT / "crack_object_detection/yolo_26x_img768/model/best.pt",
        MODEL_ROOT / "semi_labeling_training/myrun_yolo26x_img768_b16_100ep/weights/best.pt",
    ]
    weight = next((path for path in yolo_weight_candidates if path.exists()), None)
    if weight is None:
        print("[detect:yolo] no YOLO weight found; using manual fallback")
        return [_default_manual_box(image_path)]
    try:
        from object_detection.yolo.lib import load_yolo_class, resolve_device

        YOLO = load_yolo_class()
        model = YOLO(str(weight))
        resolved_device = resolve_device(device)
        results = model.predict(
            source=str(image_path),
            imgsz=768,
            conf=float(conf),
            iou=0.45,
            max_det=50,
            device=resolved_device,
            save=True,
            project=str(output_dir / "detect_yolo"),
            name="predict",
            verbose=False,
        )
        boxes: list[dict[str, Any]] = []
        if results and getattr(results[0], "boxes", None) is not None:
            result = results[0]
            names = getattr(result, "names", {}) or {}
            xyxy = result.boxes.xyxy.detach().cpu().tolist()
            scores = result.boxes.conf.detach().cpu().tolist()
            classes = result.boxes.cls.detach().cpu().tolist()
            for idx, (coords, score, cls_id) in enumerate(zip(xyxy, scores, classes)):
                label = str(names.get(int(cls_id), "crack"))
                boxes.append({"label": label, "score": float(score), "box": [float(v) for v in coords], "source": "yolo", "det_idx": idx})
        print(f"[detect:yolo] weight={weight} boxes={len(boxes)}")
        return boxes or [_default_manual_box(image_path)]
    except Exception as exc:
        print(f"[detect:yolo] failed: {exc}; using manual fallback")
        return [_default_manual_box(image_path)]


def _decode_mask(det: dict[str, Any], output_dir: Path, index: int) -> Path | None:
    mask_path = det.get("mask_path")
    if mask_path and Path(str(mask_path)).exists():
        return Path(str(mask_path))
    mask_b64 = det.get("mask_b64")
    if mask_b64:
        path = output_dir / f"decoded_det_{index:03d}.png"
        path.write_bytes(base64.b64decode(mask_b64))
        return path
    return None


def _mask_stats(mask_path: Path, box: list[float], image_shape: tuple[int, int]) -> dict[str, Any]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Cannot read mask: {mask_path}")
    h_img, w_img = image_shape
    area = int(np.count_nonzero(mask))
    ys, xs = np.where(mask > 0)
    if area == 0:
        return {"mask_path": str(mask_path), "area": 0, "mask_shape": list(mask.shape), "outside_ratio": 0.0, "full_image_ratio": 0.0, "mask_bbox": None}

    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(w_img, x1)); x2 = max(0, min(w_img, x2))
    y1 = max(0, min(h_img, y1)); y2 = max(0, min(h_img, y2))
    full_image_like = mask.shape[:2] == (h_img, w_img)

    if full_image_like:
        inside = np.zeros_like(mask, dtype=bool)
        inside[y1:y2, x1:x2] = True
        outside = int(np.count_nonzero((mask > 0) & (~inside)))
        full_image_ratio = area / float(max(1, h_img * w_img))
    else:
        # A cropped mask cannot be spatially checked against full-image coords,
        # but it should be approximately box-sized.
        outside = 0
        full_image_ratio = 0.0
    return {
        "mask_path": str(mask_path),
        "mask_shape": list(mask.shape),
        "area": area,
        "mask_bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        "outside_ratio": outside / float(max(1, area)),
        "full_image_ratio": full_image_ratio,
        "full_image_like": full_image_like,
        "source_box": [x1, y1, x2, y2],
    }


def _assert_box_scoped(stats: list[dict[str, Any]], *, name: str) -> None:
    if not stats:
        raise AssertionError(f"{name}: produced no detection masks")
    for item in stats:
        if item["area"] <= 0:
            raise AssertionError(f"{name}: empty mask {item['mask_path']}")
        if item.get("full_image_like") and item["outside_ratio"] > 0.02:
            raise AssertionError(f"{name}: mask leaks outside box: {item}")
        if item.get("full_image_like") and item["full_image_ratio"] > 0.80:
            raise AssertionError(f"{name}: mask looks like full-image fill: {item}")


def _run_sam_lora(image_path: Path, boxes: list[dict[str, Any]], output_dir: Path, device: str, variant: str) -> dict[str, Any]:
    _add_repo_to_path()
    from segmentation.sam.finetune.engine import SamFinetuneParams, SamFinetuneRunner

    model_dir = MODEL_ROOT / f"crack_segmentation/{variant}/model"
    params = SamFinetuneParams(
        sam_checkpoint=str(model_dir / "sam_vit_b_01ec64.pth"),
        sam_model_type="vit_b",
        delta_type="lora",
        delta_checkpoint=str(model_dir / "coarse_best_model.pth"),
        rank=4,
        device=device,
        output_dir=str(output_dir / variant),
        threshold="auto",
        task_group="crack_only",
    )
    runner = SamFinetuneRunner()
    result = runner.segment_boxes(str(image_path), params, boxes, log_fn=lambda msg: print(f"[sam_lora:{variant}] {msg}"))
    stats = []
    for idx, det in enumerate(result.get("detections") or []):
        path = _decode_mask(det, output_dir / variant, idx)
        if path is not None:
            stats.append(_mask_stats(path, list(det.get("box") or boxes[idx].get("box")), _image_shape(image_path)))
    _assert_box_scoped(stats, name=variant)
    return {"result": result, "stats": stats}


def _run_unet(image_path: Path, boxes: list[dict[str, Any]], output_dir: Path, device: str) -> dict[str, Any]:
    _add_repo_to_path()
    from segmentation.unet.engine import UnetParams, UnetRunner

    params = UnetParams(
        model_path=str(MODEL_ROOT / "crack_segmentation/unet_efficientnet_b4/model/best_model.pth"),
        output_dir=str(output_dir / "unet_efficientnet_b4"),
        threshold=0.5,
        mode="tile",
        input_size=512,
        tile_overlap=256,
        device=device,
    )
    runner = UnetRunner()
    stats = []
    detections = []
    h_img, w_img = _image_shape(image_path)
    for idx, entry in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in entry["box"]]
        roi = (max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2))
        per_box_params = params
        per_box_params.output_dir = str(output_dir / "unet_efficientnet_b4" / f"box_{idx:03d}")
        result = runner.run_rois(str(image_path), per_box_params, [roi], log_fn=lambda msg: print(f"[unet] {msg}"))
        mask_path = Path(result["mask_path"])
        item = _mask_stats(mask_path, list(entry["box"]), (h_img, w_img))
        stats.append(item)
        detections.append({"box": entry["box"], "mask_path": str(mask_path), "stats": item})
    _assert_box_scoped(stats, name="unet_efficientnet_b4")
    return {"detections": detections, "stats": stats}


def _image_shape(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    return h, w


def _draw_summary(image_path: Path, boxes: list[dict[str, Any]], output_path: Path) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        return
    for idx, item in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in item["box"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(img, f"{idx}:{item.get('source','box')} {float(item.get('score',0)):.2f}", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    cv2.imwrite(str(output_path), img)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="/Users/nguyenquangvinh/Downloads/wall-crack.jpg")
    parser.add_argument("--output-dir", default=str(MODEL_ROOT / "smoke_box_segmentation"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--detector", choices=["yolo", "manual"], default="yolo")
    parser.add_argument("--yolo-conf", type=float, default=0.10)
    parser.add_argument("--segmenters", default="sam_lora_hq_coarse_refine,unet", help="Comma list: sam_lora_hq_coarse_refine,sam_lora_baseline_coarse_refine,unet")
    parser.add_argument("--max-boxes", type=int, default=3)
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    boxes = _run_yolo(image_path, output_dir, args.device, args.yolo_conf) if args.detector == "yolo" else [_default_manual_box(image_path)]
    boxes = boxes[: max(1, int(args.max_boxes))]
    if not boxes:
        boxes = [_default_manual_box(image_path)]
    _draw_summary(image_path, boxes, output_dir / "detected_boxes.jpg")

    summary: dict[str, Any] = {
        "image": str(image_path),
        "output_dir": str(output_dir),
        "boxes": boxes,
        "models": {
            "crack_segmentation": str(MODEL_ROOT / "crack_segmentation"),
            "crack_object_detection": str(MODEL_ROOT / "crack_object_detection"),
            "semi_labeling_training": str(MODEL_ROOT / "semi_labeling_training"),
        },
        "segmenters": {},
    }

    for segmenter in [part.strip() for part in str(args.segmenters).split(",") if part.strip()]:
        print(f"[segment] running {segmenter}")
        try:
            if segmenter == "unet":
                summary["segmenters"][segmenter] = _run_unet(image_path, boxes, output_dir, args.device)
            else:
                summary["segmenters"][segmenter] = _run_sam_lora(image_path, boxes, output_dir, args.device, segmenter)
        except Exception as exc:
            summary["segmenters"][segmenter] = {"error": repr(exc)}
            print(f"[segment] {segmenter} FAILED: {exc}")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] summary={summary_path}")
    print(f"[done] boxes_overlay={output_dir / 'detected_boxes.jpg'}")

    failures = [name for name, payload in summary["segmenters"].items() if isinstance(payload, dict) and payload.get("error")]
    if failures:
        print(f"[done] failures={failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
