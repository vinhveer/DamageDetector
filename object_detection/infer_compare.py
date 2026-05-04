from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from object_detection.yolo.lib import load_yolo_class, resolve_device


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = {0: "crack"}
MAP_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]


@dataclass
class Box:
    image_id: str
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float | None = None
    source: str = "gt"
    index: int = 0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run/compare crack object detectors with numeric overlays.")
    parser.add_argument("--dataset-root", required=True, help="Dataset split root containing images/ and labels/.")
    parser.add_argument("--output-root", required=True, help="Root directory for infer_results outputs.")
    parser.add_argument("--run-yolo", action="store_true", help="Run YOLO inference with Ultralytics.")
    parser.add_argument("--yolo-weight", default="", help="Path to YOLO best.pt.")
    parser.add_argument("--yolo-imgsz", type=int, default=1024)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--yolo-device", default="auto")
    parser.add_argument("--yolo-batch", type=int, default=1)
    parser.add_argument("--stable-dino-coco-json", default="", help="COCO detection JSON from StableDINO eval.")
    parser.add_argument("--stable-dino-test-coco", default="", help="COCO ground-truth JSON used by StableDINO eval.")
    parser.add_argument("--stable-dino-conf", type=float, default=0.25, help="Confidence threshold applied to StableDINO COCO predictions.")
    parser.add_argument("--stable-dino-total-time", type=float, default=0.0, help="StableDINO total inference seconds from eval log.")
    parser.add_argument("--primary-iou", type=float, default=0.5)
    parser.add_argument("--side-by-side-limit", type=int, default=0, help="0 saves all side-by-side overlays.")
    return parser


def list_images(dataset_root: Path) -> list[Path]:
    image_dir = dataset_root / "images"
    return sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def clamp_box(box: Box, width: int, height: int) -> Box:
    return Box(
        image_id=box.image_id,
        class_id=box.class_id,
        x1=max(0.0, min(float(width), box.x1)),
        y1=max(0.0, min(float(height), box.y1)),
        x2=max(0.0, min(float(width), box.x2)),
        y2=max(0.0, min(float(height), box.y2)),
        conf=box.conf,
        source=box.source,
        index=box.index,
    )


def load_gt(dataset_root: Path, image_paths: list[Path]) -> tuple[dict[str, list[Box]], dict[str, dict[str, Any]]]:
    labels_dir = dataset_root / "labels"
    gt: dict[str, list[Box]] = {}
    image_info: dict[str, dict[str, Any]] = {}
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        image_id = image_path.stem
        image_info[image_id] = {"path": str(image_path), "width": width, "height": height}
        label_path = labels_dir / f"{image_id}.txt"
        boxes: list[Box] = []
        if label_path.exists():
            for raw_line in label_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_id = int(float(parts[0]))
                xc, yc, bw, bh = (float(value) for value in parts[1:5])
                x1 = (xc - bw / 2.0) * width
                y1 = (yc - bh / 2.0) * height
                x2 = (xc + bw / 2.0) * width
                y2 = (yc + bh / 2.0) * height
                box = clamp_box(Box(image_id, class_id, x1, y1, x2, y2, source="gt", index=len(boxes)), width, height)
                if box.area > 0:
                    boxes.append(box)
        gt[image_id] = boxes
    return gt, image_info


def iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def run_yolo(image_paths: list[Path], image_info: dict[str, dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, list[Box]], dict[str, float]]:
    YOLO = load_yolo_class()
    model = YOLO(str(Path(args.yolo_weight).expanduser().resolve()))
    device = resolve_device(args.yolo_device)
    predictions: dict[str, list[Box]] = {}
    times: dict[str, float] = {}
    for image_path in image_paths:
        image_id = image_path.stem
        started = time.perf_counter()
        results = model.predict(
            source=str(image_path),
            imgsz=int(args.yolo_imgsz),
            conf=float(args.yolo_conf),
            iou=float(args.yolo_iou),
            max_det=300,
            batch=int(args.yolo_batch),
            device=device,
            save=False,
            verbose=False,
        )
        times[image_id] = time.perf_counter() - started
        boxes: list[Box] = []
        if results and getattr(results[0], "boxes", None) is not None:
            result = results[0]
            xyxy = result.boxes.xyxy.detach().cpu().tolist()
            conf = result.boxes.conf.detach().cpu().tolist()
            cls = result.boxes.cls.detach().cpu().tolist()
            width = int(image_info[image_id]["width"])
            height = int(image_info[image_id]["height"])
            for idx, (coords, score, class_id) in enumerate(zip(xyxy, conf, cls)):
                box = clamp_box(
                    Box(image_id, int(class_id), coords[0], coords[1], coords[2], coords[3], float(score), "prediction", idx),
                    width,
                    height,
                )
                if box.area > 0:
                    boxes.append(box)
        predictions[image_id] = boxes
    return predictions, times


def load_stable_dino_predictions(coco_json: Path, image_info: dict[str, dict[str, Any]], test_coco: Path | None, conf_threshold: float) -> dict[str, list[Box]]:
    image_id_to_stem: dict[int, str] = {}
    if test_coco is not None and test_coco.exists():
        payload = json.loads(test_coco.read_text(encoding="utf-8"))
        for item in payload.get("images", []):
            image_id_to_stem[int(item["id"])] = Path(str(item["file_name"])).stem
    else:
        for idx, image_id in enumerate(sorted(image_info), start=1):
            image_id_to_stem[idx] = image_id

    predictions: dict[str, list[Box]] = {image_id: [] for image_id in image_info}
    payload = json.loads(coco_json.read_text(encoding="utf-8"))
    for item in payload:
        image_id = image_id_to_stem.get(int(item.get("image_id", -1)))
        if image_id is None or image_id not in image_info:
            continue
        x, y, w, h = (float(value) for value in item.get("bbox", [0, 0, 0, 0]))
        class_id = int(item.get("category_id", 1)) - 1
        score = float(item.get("score", 0.0))
        if score < conf_threshold:
            continue
        info = image_info[image_id]
        box = clamp_box(
            Box(image_id, class_id, x, y, x + w, y + h, score, "prediction", len(predictions[image_id])),
            int(info["width"]),
            int(info["height"]),
        )
        if box.area > 0:
            predictions[image_id].append(box)
    for boxes in predictions.values():
        boxes.sort(key=lambda item: float(item.conf or 0.0), reverse=True)
        for idx, box in enumerate(boxes):
            box.index = idx
    return predictions


def match_image(gt_boxes: list[Box], pred_boxes: list[Box], threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matched_gt: set[int] = set()
    pred_records: list[dict[str, Any]] = []
    for pred in sorted(pred_boxes, key=lambda item: float(item.conf or 0.0), reverse=True):
        best_iou = 0.0
        best_gt_idx: int | None = None
        for gt in gt_boxes:
            if gt.index in matched_gt or gt.class_id != pred.class_id:
                continue
            value = iou(pred, gt)
            if value > best_iou:
                best_iou = value
                best_gt_idx = gt.index
        is_tp = best_gt_idx is not None and best_iou >= threshold
        if is_tp:
            matched_gt.add(int(best_gt_idx))
        pred_records.append({"box": pred, "status": "TP" if is_tp else "FP", "best_iou": best_iou, "matched_gt_id": best_gt_idx})

    gt_records: list[dict[str, Any]] = []
    for gt in gt_boxes:
        best_iou = 0.0
        best_pred_idx: int | None = None
        for pred in pred_boxes:
            if pred.class_id != gt.class_id:
                continue
            value = iou(gt, pred)
            if value > best_iou:
                best_iou = value
                best_pred_idx = pred.index
        gt_records.append({"box": gt, "status": "matched_gt" if gt.index in matched_gt else "FN", "best_iou": best_iou, "matched_pred_id": best_pred_idx})
    return pred_records, gt_records


def compute_ap(gt_by_image: dict[str, list[Box]], pred_by_image: dict[str, list[Box]], threshold: float) -> float:
    total_gt = sum(len(boxes) for boxes in gt_by_image.values())
    if total_gt == 0:
        return 0.0
    all_preds = [box for boxes in pred_by_image.values() for box in boxes]
    all_preds.sort(key=lambda item: float(item.conf or 0.0), reverse=True)
    matched: dict[str, set[int]] = defaultdict(set)
    tp_values: list[int] = []
    fp_values: list[int] = []
    for pred in all_preds:
        best_iou = 0.0
        best_gt_idx: int | None = None
        for gt in gt_by_image.get(pred.image_id, []):
            if gt.index in matched[pred.image_id] or gt.class_id != pred.class_id:
                continue
            value = iou(pred, gt)
            if value > best_iou:
                best_iou = value
                best_gt_idx = gt.index
        if best_gt_idx is not None and best_iou >= threshold:
            matched[pred.image_id].add(best_gt_idx)
            tp_values.append(1)
            fp_values.append(0)
        else:
            tp_values.append(0)
            fp_values.append(1)
    if not tp_values:
        return 0.0
    cum_tp = 0
    cum_fp = 0
    recalls = [0.0]
    precisions = [1.0]
    for tp, fp in zip(tp_values, fp_values):
        cum_tp += tp
        cum_fp += fp
        recalls.append(cum_tp / total_gt)
        precisions.append(cum_tp / max(cum_tp + cum_fp, 1))
    recalls.append(1.0)
    precisions.append(0.0)
    for idx in range(len(precisions) - 2, -1, -1):
        precisions[idx] = max(precisions[idx], precisions[idx + 1])
    ap = 0.0
    for idx in range(1, len(recalls)):
        if recalls[idx] != recalls[idx - 1]:
            ap += (recalls[idx] - recalls[idx - 1]) * precisions[idx]
    return ap


def evaluate_detector(name: str, gt_by_image: dict[str, list[Box]], pred_by_image: dict[str, list[Box]], image_info: dict[str, dict[str, Any]], times: dict[str, float], threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]]:
    image_rows: list[dict[str, Any]] = []
    matches: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    tp = fp = fn = 0
    for image_id in sorted(image_info):
        pred_records, gt_records = match_image(gt_by_image.get(image_id, []), pred_by_image.get(image_id, []), threshold)
        matches[image_id] = (pred_records, gt_records)
        image_tp = sum(1 for item in pred_records if item["status"] == "TP")
        image_fp = sum(1 for item in pred_records if item["status"] == "FP")
        image_fn = sum(1 for item in gt_records if item["status"] == "FN")
        tp += image_tp
        fp += image_fp
        fn += image_fn
        precision = image_tp / (image_tp + image_fp) if image_tp + image_fp else 0.0
        recall = image_tp / (image_tp + image_fn) if image_tp + image_fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        best_ious = [float(item["best_iou"]) for item in pred_records] + [float(item["best_iou"]) for item in gt_records]
        avg_best_iou = sum(best_ious) / len(best_ious) if best_ious else 0.0
        infer_time = float(times.get(image_id, 0.0))
        image_rows.append(
            {
                "detector": name,
                "image_id": image_id,
                "image_path": image_info[image_id]["path"],
                "width": image_info[image_id]["width"],
                "height": image_info[image_id]["height"],
                "gt_count": len(gt_by_image.get(image_id, [])),
                "pred_count": len(pred_by_image.get(image_id, [])),
                "tp": image_tp,
                "fp": image_fp,
                "fn": image_fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "avg_best_iou": avg_best_iou,
                "inference_time_ms": infer_time * 1000.0,
                "fps": 1.0 / infer_time if infer_time > 0 else 0.0,
            }
        )

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ap_by_threshold = {f"ap_{int(t * 100)}": compute_ap(gt_by_image, pred_by_image, t) for t in MAP_THRESHOLDS}
    total_time = sum(float(value) for value in times.values())
    num_images = len(image_info)
    summary = {
        "detector": name,
        "num_images": num_images,
        "num_gt_boxes": sum(len(boxes) for boxes in gt_by_image.values()),
        "num_predictions": sum(len(boxes) for boxes in pred_by_image.values()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map_50": ap_by_threshold["ap_50"],
        "map_50_95": sum(ap_by_threshold.values()) / len(ap_by_threshold),
        "avg_gt_boxes_per_image": sum(len(boxes) for boxes in gt_by_image.values()) / max(num_images, 1),
        "avg_pred_boxes_per_image": sum(len(boxes) for boxes in pred_by_image.values()) / max(num_images, 1),
        "total_time_seconds": total_time,
        "avg_time_ms": total_time * 1000.0 / max(num_images, 1) if total_time else 0.0,
        "fps": num_images / total_time if total_time else 0.0,
        **ap_by_threshold,
    }
    return summary, image_rows, matches


def put_text(image: Any, text: str, origin: tuple[int, int], scale: float = 0.45, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_overlay(detector: str, image_path: Path, gt_records: list[dict[str, Any]], pred_records: list[dict[str, Any]], summary: dict[str, Any], output_path: Path, threshold: float) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    height, width = image.shape[:2]
    header_h = 118
    cv2.rectangle(image, (0, 0), (width, header_h), (30, 30, 30), -1)
    header_lines = [
        f"Model: {detector} | Image: {image_path.name} | Size: {width}x{height}",
        f"GT={len(gt_records)} Pred={len(pred_records)} TP={summary['tp']} FP={summary['fp']} FN={summary['fn']} IoU={threshold:.2f}",
        f"P={summary['precision']:.3f} R={summary['recall']:.3f} F1={summary['f1']:.3f} mAP50={summary['map_50']:.3f} mAP50-95={summary['map_50_95']:.3f}",
        "GT: green/orange FN | Pred: blue TP, red FP",
    ]
    y = 22
    for line in header_lines:
        put_text(image, line, (10, y), scale=0.48)
        y += 27

    for item in gt_records:
        box: Box = item["box"]
        status = str(item["status"])
        color = (0, 165, 255) if status == "FN" else (0, 255, 0)
        x1, y1, x2, y2 = (int(round(value)) for value in (box.x1, box.y1, box.x2, box.y2))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"GT#{box.index} {status} IoU={float(item['best_iou']):.2f} [{x1},{y1},{x2},{y2}] A={box.area:.0f}"
        put_text(image, label, (x1, max(15, y1 - 8)), scale=0.36, color=color)

    for item in pred_records:
        box: Box = item["box"]
        status = str(item["status"])
        color = (255, 80, 0) if status == "TP" else (0, 0, 255)
        x1, y1, x2, y2 = (int(round(value)) for value in (box.x1, box.y1, box.x2, box.y2))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = (
            f"P#{box.index} crack {float(box.conf or 0):.2f} {status} IoU={float(item['best_iou']):.2f} "
            f"[{x1},{y1},{x2},{y2}] {int(box.width)}x{int(box.height)} A={box.area:.0f}"
        )
        put_text(image, label, (x1, min(height - 8, max(15, y2 + 15))), scale=0.36, color=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def detection_rows(detector: str, gt_records_by_image: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]], image_info: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image_id in sorted(gt_records_by_image):
        width = float(image_info[image_id]["width"])
        height = float(image_info[image_id]["height"])
        pred_records, gt_records = gt_records_by_image[image_id]
        for item in pred_records:
            box: Box = item["box"]
            rows.append(_box_row(detector, box, "prediction", width, height, item))
        for item in gt_records:
            box = item["box"]
            rows.append(_box_row(detector, box, "gt", width, height, item))
    return rows


def _box_row(detector: str, box: Box, source: str, width: float, height: float, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "detector": detector,
        "image_id": box.image_id,
        "source": source,
        "class_id": box.class_id,
        "class_name": CLASS_NAMES.get(box.class_id, str(box.class_id)),
        "confidence": "" if box.conf is None else float(box.conf),
        "x1": box.x1,
        "y1": box.y1,
        "x2": box.x2,
        "y2": box.y2,
        "x_center_norm": ((box.x1 + box.x2) / 2.0) / width if width else 0.0,
        "y_center_norm": ((box.y1 + box.y2) / 2.0) / height if height else 0.0,
        "width_norm": box.width / width if width else 0.0,
        "height_norm": box.height / height if height else 0.0,
        "area_px": box.area,
        "matched_gt_id": item.get("matched_gt_id", ""),
        "matched_pred_id": item.get("matched_pred_id", ""),
        "best_iou": item.get("best_iou", 0.0),
        "status": item.get("status", ""),
        "rank_by_confidence": box.index if source == "prediction" else "",
    }


def save_side_by_side(detector_outputs: dict[str, Path], image_ids: list[str], output_dir: Path, limit: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = image_ids if limit <= 0 else image_ids[:limit]
    for image_id in selected:
        paths = [detector_outputs[name] / f"{image_id}.jpg" for name in detector_outputs]
        images = [cv2.imread(str(path)) for path in paths if path.exists()]
        if len(images) < 2:
            continue
        min_h = min(image.shape[0] for image in images)
        resized = [cv2.resize(image, (int(image.shape[1] * min_h / image.shape[0]), min_h)) for image in images]
        combined = cv2.hconcat(resized)
        cv2.imwrite(str(output_dir / f"{image_id}.jpg"), combined)


def main() -> int:
    args = build_parser().parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    image_paths = list_images(dataset_root)
    gt_by_image, image_info = load_gt(dataset_root, image_paths)

    detectors: dict[str, tuple[dict[str, list[Box]], dict[str, float], dict[str, Any]]] = {}
    if args.run_yolo:
        yolo_predictions, yolo_times = run_yolo(image_paths, image_info, args)
        detectors["yolo_crack_500"] = (yolo_predictions, yolo_times, {"conf": args.yolo_conf, "iou": args.yolo_iou, "imgsz": args.yolo_imgsz})
    if args.stable_dino_coco_json:
        stable_json = Path(args.stable_dino_coco_json).expanduser().resolve()
        test_coco = Path(args.stable_dino_test_coco).expanduser().resolve() if args.stable_dino_test_coco else None
        stable_predictions = load_stable_dino_predictions(stable_json, image_info, test_coco, float(args.stable_dino_conf))
        stable_times = {}
        if float(args.stable_dino_total_time or 0.0) > 0.0 and image_info:
            per_image_time = float(args.stable_dino_total_time) / len(image_info)
            stable_times = {image_id: per_image_time for image_id in image_info}
        detectors["stable_dino_crack_500"] = (
            stable_predictions,
            stable_times,
            {"source_json": str(stable_json), "conf": float(args.stable_dino_conf)},
        )

    comparison_rows: list[dict[str, Any]] = []
    comparison_summary = {
        "dataset": {"path": str(dataset_root), "num_images": len(image_info), "num_gt_boxes": sum(len(v) for v in gt_by_image.values()), "classes": CLASS_NAMES},
        "iou_primary": float(args.primary_iou),
        "map_thresholds": MAP_THRESHOLDS,
        "detectors": [],
    }
    overlay_dirs: dict[str, Path] = {}
    image_ids = sorted(image_info)
    for detector_name, (predictions, times, run_config) in detectors.items():
        summary, image_rows, matches = evaluate_detector(detector_name, gt_by_image, predictions, image_info, times, float(args.primary_iou))
        summary["run_config"] = run_config
        model_dir = output_root / detector_name
        overlay_dir = model_dir / "overlays"
        overlay_dirs[detector_name] = overlay_dir
        for row in image_rows:
            image_id = str(row["image_id"])
            row["overlay_path"] = str(overlay_dir / f"{image_id}.jpg")
            pred_records, gt_records = matches[image_id]
            draw_overlay(detector_name, Path(image_info[image_id]["path"]), gt_records, pred_records, summary, overlay_dir / f"{image_id}.jpg", float(args.primary_iou))
        det_rows = detection_rows(detector_name, matches, image_info)
        write_csv(model_dir / "image_metrics.csv", image_rows, list(image_rows[0].keys()) if image_rows else [])
        write_csv(
            model_dir / "predictions.csv",
            det_rows,
            [
                "detector",
                "image_id",
                "source",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "x_center_norm",
                "y_center_norm",
                "width_norm",
                "height_norm",
                "area_px",
                "matched_gt_id",
                "matched_pred_id",
                "best_iou",
                "status",
                "rank_by_confidence",
            ],
        )
        (model_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        comparison_rows.append(summary)
        comparison_summary["detectors"].append(summary)

    comparison_dir = output_root / "comparison"
    if len(overlay_dirs) >= 2:
        save_side_by_side(overlay_dirs, image_ids, comparison_dir / "side_by_side", int(args.side_by_side_limit))
    write_csv(
        comparison_dir / "metrics_summary.csv",
        comparison_rows,
        [
            "detector",
            "num_images",
            "num_gt_boxes",
            "num_predictions",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "map_50",
            "map_50_95",
            "avg_gt_boxes_per_image",
            "avg_pred_boxes_per_image",
            "total_time_seconds",
            "avg_time_ms",
            "fps",
        ],
    )
    (comparison_dir / "comparison_summary.json").write_text(json.dumps(comparison_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparison_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
