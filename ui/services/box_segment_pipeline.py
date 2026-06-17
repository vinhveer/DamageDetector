from __future__ import annotations

import base64
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT.parent
MODEL_ROOT = LAB / "model_with_inference"


LogFn = Callable[[str], None] | None


@dataclass(frozen=True)
class BoxSegmentModels:
    crack_segmentation: Path = MODEL_ROOT / "crack_segmentation"
    crack_object_detection: Path = MODEL_ROOT / "crack_object_detection"
    semi_labeling_training: Path = MODEL_ROOT / "semi_labeling_training"

    @property
    def yolo_weights(self) -> list[Path]:
        return [
            self.crack_object_detection / "yolo_26x_img768/model/best.pt",
            self.semi_labeling_training / "myrun_yolo26x_img768_b16_100ep/weights/best.pt",
        ]

    @property
    def unet_weight(self) -> Path:
        return self.crack_segmentation / "unet_efficientnet_b4/model/best_model.pth"


def add_repo_to_path() -> None:
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def image_shape(image_path: str | Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(str(image_path))
    h, w = image.shape[:2]
    return int(h), int(w)


def default_manual_box(image_path: str | Path) -> dict[str, Any]:
    h, w = image_shape(image_path)
    return {
        "label": "crack",
        "score": 1.0,
        "box": [max(0, int(w * 0.08)), max(0, int(h * 0.02)), min(w - 1, int(w * 0.46)), min(h - 1, int(h * 0.98))],
        "source": "manual_fallback",
        "detector_name": "Manual",
    }


def clamp_box(box: list[float] | tuple[float, ...], image_hw: tuple[int, int]) -> tuple[int, int, int, int] | None:
    h, w = image_hw
    if len(box) != 4:
        return None
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def run_yolo_detection(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    models: BoxSegmentModels | None = None,
    weight_path: str | Path | None = None,
    device: str = "auto",
    conf: float = 0.10,
    iou: float = 0.45,
    imgsz: int = 768,
    max_dets: int = 50,
    fallback: bool = True,
    log_fn: LogFn = None,
) -> list[dict[str, Any]]:
    add_repo_to_path()
    image_path = Path(image_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    models = models or BoxSegmentModels()
    candidates = [Path(weight_path).expanduser()] if weight_path else models.yolo_weights
    weight = next((path for path in candidates if path.exists()), None)
    if weight is None:
        if log_fn:
            log_fn("YOLO weight not found; using manual fallback box.")
        return [default_manual_box(image_path)] if fallback else []
    try:
        from object_detection.yolo.lib import load_yolo_class, resolve_device

        YOLO = load_yolo_class()
        model = YOLO(str(weight))
        resolved_device = resolve_device(device)
        results = model.predict(
            source=str(image_path),
            imgsz=int(imgsz),
            conf=float(conf),
            iou=float(iou),
            max_det=int(max_dets),
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
                boxes.append(
                    {
                        "label": label,
                        "score": float(score),
                        "box": [float(v) for v in coords],
                        "source": "yolo",
                        "detector_name": "YOLO",
                        "model_name": "YOLO",
                        "det_idx": idx,
                    }
                )
        if log_fn:
            log_fn(f"YOLO weight={weight} boxes={len(boxes)}")
        return boxes or ([default_manual_box(image_path)] if fallback else [])
    except Exception as exc:
        if log_fn:
            log_fn(f"YOLO failed: {exc}; using manual fallback box.")
        return [default_manual_box(image_path)] if fallback else []


def decode_mask_path(det: dict[str, Any], output_dir: str | Path, index: int) -> Path | None:
    mask_path = det.get("mask_path")
    if mask_path and Path(str(mask_path)).exists():
        return Path(str(mask_path))
    mask_b64 = det.get("mask_b64")
    if mask_b64:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"decoded_det_{index:03d}.png"
        path.write_bytes(base64.b64decode(str(mask_b64)))
        return path
    return None


def mask_stats(mask_path: str | Path, box: list[float], image_hw: tuple[int, int]) -> dict[str, Any]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Cannot read mask: {mask_path}")
    h_img, w_img = image_hw
    area = int(np.count_nonzero(mask))
    ys, xs = np.where(mask > 0)
    full_image_like = mask.shape[:2] == (h_img, w_img)
    if area == 0:
        return {
            "mask_path": str(mask_path),
            "area_px": 0,
            "mask_shape": list(mask.shape),
            "outside_px": 0,
            "outside_ratio": 0.0,
            "full_image_ratio": 0.0,
            "full_image_like": full_image_like,
            "mask_bbox": None,
        }
    clamped = clamp_box(box, image_hw)
    if clamped is None:
        raise RuntimeError(f"Invalid source box: {box}")
    x1, y1, x2, y2 = clamped
    if full_image_like:
        inside = np.zeros_like(mask, dtype=bool)
        inside[y1:y2, x1:x2] = True
        outside_px = int(np.count_nonzero((mask > 0) & (~inside)))
        full_image_ratio = area / float(max(1, h_img * w_img))
    else:
        outside_px = 0
        full_image_ratio = 0.0
    return {
        "mask_path": str(mask_path),
        "mask_shape": list(mask.shape),
        "area_px": area,
        "mask_bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        "outside_px": outside_px,
        "outside_ratio": outside_px / float(max(1, area)),
        "full_image_ratio": full_image_ratio,
        "full_image_like": full_image_like,
        "source_box": [x1, y1, x2, y2],
        "box_area_px": int((x2 - x1) * (y2 - y1)),
        "box_fill_ratio": area / float(max(1, (x2 - x1) * (y2 - y1))),
    }


def assert_box_scoped(stats: list[dict[str, Any]], *, name: str, max_outside_ratio: float = 0.02, max_full_image_ratio: float = 0.80) -> None:
    if not stats:
        raise AssertionError(f"{name}: produced no detection masks")
    for item in stats:
        if int(item["area_px"]) <= 0:
            raise AssertionError(f"{name}: empty mask {item['mask_path']}")
        if item.get("full_image_like") and float(item["outside_ratio"]) > max_outside_ratio:
            raise AssertionError(f"{name}: mask leaks outside box: {item}")
        if item.get("full_image_like") and float(item["full_image_ratio"]) > max_full_image_ratio:
            raise AssertionError(f"{name}: mask looks like full-image fill: {item}")


def run_unet_boxes(
    image_path: str | Path,
    boxes: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    models: BoxSegmentModels | None = None,
    model_path: str | Path | None = None,
    device: str = "auto",
    threshold: float = 0.5,
    input_size: int = 512,
    tile_overlap: int = 256,
    apply_postprocessing: bool = True,
    log_fn: LogFn = None,
) -> dict[str, Any]:
    add_repo_to_path()
    from segmentation.unet.engine import UnetParams, UnetRunner

    image_path = Path(image_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    models = models or BoxSegmentModels()
    weight = Path(model_path).expanduser() if model_path else models.unet_weight
    params = UnetParams(
        model_path=str(weight),
        output_dir=str(output_dir),
        threshold=float(threshold),
        apply_postprocessing=bool(apply_postprocessing),
        mode="tile",
        input_size=int(input_size),
        tile_overlap=int(tile_overlap),
        device=str(device),
    )
    runner = UnetRunner()
    image_hw = image_shape(image_path)
    detections: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    h_img, w_img = image_hw
    for idx, entry in enumerate(boxes):
        box = entry.get("box")
        roi = clamp_box(box, image_hw) if isinstance(box, (list, tuple)) else None
        if roi is None:
            continue
        x1, y1, x2, y2 = roi
        per_box_params = params
        object.__setattr__(per_box_params, "output_dir", str(output_dir / "unet_efficientnet_b4" / f"box_{idx:03d}"))
        result = runner.run_rois(
            str(image_path),
            per_box_params,
            [(x1, y1, x2, y2)],
            log_fn=(lambda msg, i=idx: log_fn(f"UNet box {i + 1}/{len(boxes)}: {msg}") if log_fn else None),
        )
        mask_path = Path(str(result.get("mask_path") or ""))
        if not mask_path.exists():
            continue
        item_stats = mask_stats(mask_path, [float(x1), float(y1), float(x2), float(y2)], image_hw)
        stats.append(item_stats)
        detections.append(
            {
                "label": str(entry.get("label") or "crack"),
                "score": float(entry.get("score") or 0.0),
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "mask_path": str(mask_path),
                "mask_is_crop": False,
                "model_name": "UnetCrackBox",
                "detector_name": entry.get("detector_name") or entry.get("source") or "detect",
                "det_idx": entry.get("det_idx", idx),
                "stats": item_stats,
            }
        )
    return {
        "image_path": str(image_path),
        "output_dir": str(output_dir),
        "detections": detections,
        "stats": stats,
        "image_shape": [h_img, w_img],
        "model_path": str(weight),
    }


def run_sam_lora_boxes(
    image_path: str | Path,
    boxes: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    variant: str = "sam_lora_hq_coarse_refine",
    models: BoxSegmentModels | None = None,
    device: str = "auto",
    log_fn: LogFn = None,
) -> dict[str, Any]:
    add_repo_to_path()
    from segmentation.sam.finetune.engine import SamFinetuneParams, SamFinetuneRunner

    image_path = Path(image_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    models = models or BoxSegmentModels()
    model_dir = models.crack_segmentation / variant / "model"
    params = SamFinetuneParams(
        sam_checkpoint=str(model_dir / "sam_vit_b_01ec64.pth"),
        sam_model_type="vit_b",
        delta_type="lora",
        delta_checkpoint=str(model_dir / "coarse_best_model.pth"),
        rank=4,
        device=str(device),
        output_dir=str(output_dir / variant),
        threshold="auto",
        task_group="crack_only",
    )
    runner = SamFinetuneRunner()
    result = runner.segment_boxes(
        str(image_path),
        params,
        boxes,
        log_fn=(lambda msg: log_fn(f"SAM-LoRA {variant}: {msg}") if log_fn else None),
    )
    image_hw = image_shape(image_path)
    stats = []
    for idx, det in enumerate(result.get("detections") or []):
        path = decode_mask_path(det, output_dir / variant, idx)
        if path is not None:
            stats.append(mask_stats(path, list(det.get("box") or boxes[idx].get("box")), image_hw))
    payload = dict(result or {})
    payload["stats"] = stats
    payload["model_dir"] = str(model_dir)
    return payload


def draw_detection_overlay(image_path: str | Path, boxes: list[dict[str, Any]], output_path: str | Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    for idx, item in enumerate(boxes):
        box = item.get("box")
        roi = clamp_box(box, image.shape[:2]) if isinstance(box, (list, tuple)) else None
        if roi is None:
            continue
        x1, y1, x2, y2 = roi
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), 2)
        label = f"{idx}:{item.get('source', item.get('detector_name', 'box'))} {float(item.get('score', 0)):.2f}"
        cv2.putText(image, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def run_box_segment_smoke(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "auto",
    detector: str = "yolo",
    segmenters: list[str] | None = None,
    max_boxes: int = 3,
    yolo_conf: float = 0.10,
    log_fn: LogFn = None,
) -> dict[str, Any]:
    image_path = Path(image_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    models = BoxSegmentModels()
    boxes = (
        run_yolo_detection(image_path, output_dir, models=models, device=device, conf=yolo_conf, log_fn=log_fn)
        if detector == "yolo"
        else [default_manual_box(image_path)]
    )
    boxes = boxes[: max(1, int(max_boxes))] or [default_manual_box(image_path)]
    draw_detection_overlay(image_path, boxes, output_dir / "detected_boxes.jpg")
    summary: dict[str, Any] = {
        "image": str(image_path),
        "image_shape": list(image_shape(image_path)),
        "output_dir": str(output_dir),
        "boxes": boxes,
        "models": {
            "crack_segmentation": str(models.crack_segmentation),
            "crack_object_detection": str(models.crack_object_detection),
            "semi_labeling_training": str(models.semi_labeling_training),
        },
        "segmenters": {},
    }
    for segmenter in segmenters or ["sam_lora_hq_coarse_refine", "unet"]:
        try:
            if log_fn:
                log_fn(f"Running segmenter: {segmenter}")
            if segmenter == "unet":
                result = run_unet_boxes(image_path, boxes, output_dir, models=models, device=device, log_fn=log_fn)
                assert_box_scoped(list(result.get("stats") or []), name="unet")
            else:
                result = run_sam_lora_boxes(image_path, boxes, output_dir, variant=segmenter, models=models, device=device, log_fn=log_fn)
                assert_box_scoped(list(result.get("stats") or []), name=segmenter)
            summary["segmenters"][segmenter] = result
        except Exception as exc:
            summary["segmenters"][segmenter] = {"error": repr(exc)}
            if log_fn:
                log_fn(f"{segmenter} FAILED: {exc}")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
