"""Run selected HinhAnh ROI detection and render comparable overlays.

This script is intentionally project-specific for the 16 ROI images selected for
the thesis figure/comparison. It prepares a reproducible input manifest, runs
YOLO, converts detections to a common JSON/CSV format, renders overlays, and can
normalize StableDINO COCO results when they are available.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


CONF = 0.2
YOLO_IOU = 0.45
STABLE_DINO_NMS = 0.8
IMG_SIZE = 768
STABLE_DINO_NAMES = ["crack", "mold", "spall"]

GROUP_1 = "cum_1_it_ngoai_canh"
GROUP_2 = "cum_2_co_ngoai_canh"

SELECTED = [
    {"roi": 59, "group": GROUP_1, "file": "DSC01279__roi59.png"},
    {"roi": 223, "group": GROUP_1, "file": "DSC01310__roi223.png"},
    {"roi": 224, "group": GROUP_1, "file": "DSC01310__roi224.png"},
    {"roi": 228, "group": GROUP_1, "file": "DSC01310__roi228.png"},
    {"roi": 790, "group": GROUP_1, "file": "DSC01376__roi790.png"},
    {"roi": 792, "group": GROUP_1, "file": "DSC01376__roi792.png"},
    {"roi": 796, "group": GROUP_1, "file": "DSC01376__roi796.png"},
    {"roi": 797, "group": GROUP_1, "file": "DSC01376__roi797.png"},
    {"roi": 791, "group": GROUP_2, "file": "DSC01376__roi791.png"},
    {"roi": 789, "group": GROUP_2, "file": "DSC01376__roi789.png"},
    {"roi": 249, "group": GROUP_2, "file": "DSC01312__roi249.png"},
    {"roi": 247, "group": GROUP_2, "file": "DSC01312__roi247.png"},
    {"roi": 51, "group": GROUP_2, "file": "DSC01277__roi51.png"},
    {"roi": 53, "group": GROUP_2, "file": "DSC01277__roi53.png"},
    {"roi": 117, "group": GROUP_2, "file": "DSC01294__roi117.png"},
    {"roi": 118, "group": GROUP_2, "file": "DSC01294__roi118.png"},
]

COLORS = {
    "crack": (33, 150, 243),
    "spall": (0, 188, 212),
    "mold": (76, 175, 80),
}
FALLBACK_COLOR = (255, 87, 34)


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def output_root() -> Path:
    return workspace_root() / "model_with_inference" / "semi_labeling_training" / "selected_hinhanh_conf020_semilabeling"


def image_root() -> Path:
    return workspace_root() / "data" / "HinhAnh"


def yolo_weights() -> Path:
    return workspace_root() / "model_with_inference" / "semi_labeling_training" / "myrun_yolo26x_img768_b16_100ep" / "weights" / "best.pt"


def stable_dino_checkpoint() -> Path:
    return workspace_root() / "model_with_inference" / "semi_labeling_training" / "myrun_stabledino_r50_img768_b16_28600it" / "model_best.pth"


def _load_font(font_size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def _ensure_dirs() -> None:
    root = output_root()
    dirs = [
        root / "inputs" / "images",
        root / "yolo_26x_img768" / "raw",
        root / "yolo_26x_img768" / "boxes" / "per_image",
        root / "yolo_26x_img768" / "overlays" / "all",
        root / "yolo_26x_img768" / "overlays" / GROUP_1,
        root / "yolo_26x_img768" / "overlays" / GROUP_2,
        root / "yolo_26x_img768" / "logs",
        root / "stable_dino_r50_img768" / "raw",
        root / "stable_dino_r50_img768" / "boxes" / "per_image",
        root / "stable_dino_r50_img768" / "overlays" / "all",
        root / "stable_dino_r50_img768" / "overlays" / GROUP_1,
        root / "stable_dino_r50_img768" / "overlays" / GROUP_2,
        root / "stable_dino_r50_img768" / "logs",
        root / "compare" / "side_by_side",
    ]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def prepare_inputs() -> list[dict[str, Any]]:
    _ensure_dirs()
    manifest: list[dict[str, Any]] = []
    missing: list[Path] = []
    inputs_dir = output_root() / "inputs" / "images"
    for row in SELECTED:
        src = image_root() / str(row["file"])
        dst = inputs_dir / src.name
        if not src.is_file():
            missing.append(src)
            continue
        shutil.copy2(src, dst)
        item = {
            "roi": int(row["roi"]),
            "group": str(row["group"]),
            "image": src.name,
            "source_path": str(src),
            "input_path": str(dst),
        }
        manifest.append(item)
    if missing:
        raise FileNotFoundError("Missing selected images:\n" + "\n".join(str(path) for path in missing))

    inputs_root = output_root() / "inputs"
    (inputs_root / "selected_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (inputs_root / "selected_images.txt").write_text(
        "\n".join(item["input_path"] for item in manifest) + "\n", encoding="utf-8"
    )
    return manifest


def load_manifest() -> list[dict[str, Any]]:
    path = output_root() / "inputs" / "selected_manifest.json"
    if not path.is_file():
        return prepare_inputs()
    return json.loads(path.read_text(encoding="utf-8"))


def _xyxy_to_xywh(xyxy: list[float]) -> list[float]:
    x1, y1, x2, y2 = xyxy
    return [x1, y1, x2 - x1, y2 - y1]


def _draw_overlay(src_path: Path, boxes: list[dict[str, Any]], out_path: Path) -> None:
    image = Image.open(src_path).convert("RGB")
    width, _height = image.size
    draw = ImageDraw.Draw(image)
    line_w = max(2, width // 320)
    font_size = max(14, width // 45)
    font = _load_font(font_size)
    for box in sorted(boxes, key=lambda item: float(item.get("score", 0.0))):
        label = str(box.get("label", "crack"))
        score = float(box.get("score", 0.0))
        x1, y1, x2, y2 = [float(v) for v in box["xyxy"]]
        color = COLORS.get(label, FALLBACK_COLOR)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
        text = f"{label} {score:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0] + 8
        th = bbox[3] - bbox[1] + 6
        y_text = max(0, y1 - th)
        draw.rectangle([x1, y_text, x1 + tw, y_text + th], fill=color)
        draw.text((x1 + 4, y_text + 2), text, fill=(255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=92)


def _write_predictions(model_name: str, rows: list[dict[str, Any]]) -> None:
    model_root = output_root() / model_name
    boxes_root = model_root / "boxes"
    boxes_root.mkdir(parents=True, exist_ok=True)
    (boxes_root / "predictions_conf020.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    csv_path = boxes_root / "predictions_conf020.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "roi", "group", "image", "label", "score", "x1", "y1", "x2", "y2", "x", "y", "w", "h"])
        for item in rows:
            for box in item["boxes"]:
                x1, y1, x2, y2 = box["xyxy"]
                x, y, w, h = box["xywh"]
                writer.writerow([
                    item["model"], item["roi"], item["group"], item["image"], box["label"],
                    f"{float(box['score']):.6f}", f"{x1:.3f}", f"{y1:.3f}", f"{x2:.3f}", f"{y2:.3f}",
                    f"{x:.3f}", f"{y:.3f}", f"{w:.3f}", f"{h:.3f}",
                ])
    per_image = boxes_root / "per_image"
    per_image.mkdir(parents=True, exist_ok=True)
    for item in rows:
        path = per_image / f"roi{item['roi']}_{Path(item['image']).stem}.json"
        path.write_text(json.dumps(item, indent=2, ensure_ascii=False), encoding="utf-8")


def run_yolo(device: str = "auto") -> list[dict[str, Any]]:
    manifest = prepare_inputs()
    weights = yolo_weights()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    from ultralytics import YOLO

    model = YOLO(str(weights))
    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    model_root = output_root() / "yolo_26x_img768"
    for item in manifest:
        image_path = Path(item["input_path"])
        results = model.predict(
            source=str(image_path),
            imgsz=IMG_SIZE,
            conf=CONF,
            iou=YOLO_IOU,
            device=None if device == "auto" else device,
            verbose=False,
        )
        boxes: list[dict[str, Any]] = []
        for result in results:
            names = getattr(result, "names", {}) or {}
            result_boxes = getattr(result, "boxes", None)
            if result_boxes is None or getattr(result_boxes, "xyxy", None) is None:
                continue
            xyxy_rows = result_boxes.xyxy.detach().cpu().tolist()
            score_rows = result_boxes.conf.detach().cpu().tolist()
            class_rows = result_boxes.cls.detach().cpu().tolist()
            for xyxy, score, cls in zip(xyxy_rows, score_rows, class_rows):
                xyxy_f = [float(v) for v in xyxy]
                label = str(names.get(int(cls), int(cls)))
                boxes.append({
                    "label": label,
                    "score": float(score),
                    "xyxy": xyxy_f,
                    "xywh": _xyxy_to_xywh(xyxy_f),
                })
        row = {
            "roi": int(item["roi"]),
            "group": str(item["group"]),
            "image": str(item["image"]),
            "model": "yolo_26x_img768",
            "conf_threshold": CONF,
            "imgsz": IMG_SIZE,
            "iou": YOLO_IOU,
            "boxes": boxes,
        }
        rows.append(row)
        raw_rows.append({"path": str(image_path), "boxes": boxes})
        all_overlay = model_root / "overlays" / "all" / image_path.name
        group_overlay = model_root / "overlays" / str(item["group"]) / image_path.name
        _draw_overlay(image_path, boxes, all_overlay)
        shutil.copy2(all_overlay, group_overlay)
        print(f"YOLO roi{item['roi']}: {len(boxes)} boxes")

    _write_predictions("yolo_26x_img768", rows)
    raw_path = model_root / "raw" / "predictions_conf020.json"
    raw_path.write_text(json.dumps({"results": raw_rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def _find_stabledino_results(raw_root: Path) -> Path | None:
    candidates = [
        raw_root / "coco_instances_results.json",
        raw_root / "inference" / "coco_instances_results.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(raw_root.rglob("coco_instances_results.json")) if raw_root.exists() else []
    return matches[0] if matches else None


def _load_stabledino_meta(raw_root: Path) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    candidates = [
        raw_root / "dataset_cache" / "val_coco.json",
        raw_root / "dataset_cache" / "test_coco.json",
        raw_root / "image_only_coco" / "annotations" / "instances_val.json",
    ]
    candidates.extend(sorted(raw_root.rglob("*_coco.json")))
    candidates.extend(sorted(raw_root.rglob("instances_val.json")))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        data = json.loads(candidate.read_text(encoding="utf-8"))
        if "images" not in data:
            continue
        images = {int(img["id"]): img for img in data.get("images", [])}
        categories = {int(cat["id"]): str(cat.get("name", cat["id"])) for cat in data.get("categories", [])}
        return images, categories
    raise FileNotFoundError(f"No StableDINO COCO metadata found in {raw_root}")


def normalize_stabledino() -> list[dict[str, Any]]:
    manifest = load_manifest()
    by_image = {item["image"]: item for item in manifest}
    model_root = output_root() / "stable_dino_r50_img768"
    raw_root = model_root / "raw"
    result_path = _find_stabledino_results(raw_root)
    if result_path is None:
        raise FileNotFoundError(f"No coco_instances_results.json found in {raw_root}")
    id_to_image, cat_to_name = _load_stabledino_meta(raw_root)
    predictions = json.loads(result_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = {item["image"]: [] for item in manifest}
    for pred in predictions:
        score = float(pred.get("score", 0.0))
        if score < CONF:
            continue
        info = id_to_image.get(int(pred["image_id"]))
        if not info:
            continue
        image_name = Path(str(info["file_name"])).name
        if image_name not in grouped:
            continue
        x, y, w, h = [float(v) for v in pred.get("bbox", [0, 0, 0, 0])[:4]]
        label = cat_to_name.get(int(pred.get("category_id", 1)), "crack")
        grouped[image_name].append({
            "label": label,
            "score": score,
            "xyxy": [x, y, x + w, y + h],
            "xywh": [x, y, w, h],
        })
    rows: list[dict[str, Any]] = []
    for image_name, boxes in grouped.items():
        item = by_image[image_name]
        row = {
            "roi": int(item["roi"]),
            "group": str(item["group"]),
            "image": image_name,
            "model": "stable_dino_r50_img768",
            "conf_threshold": CONF,
            "imgsz": IMG_SIZE,
            "nms": STABLE_DINO_NMS,
            "boxes": boxes,
        }
        rows.append(row)
        image_path = Path(item["input_path"])
        all_overlay = model_root / "overlays" / "all" / image_name
        group_overlay = model_root / "overlays" / str(item["group"]) / image_name
        _draw_overlay(image_path, boxes, all_overlay)
        shutil.copy2(all_overlay, group_overlay)
    rows.sort(key=lambda item: int(item["roi"]))
    _write_predictions("stable_dino_r50_img768", rows)
    return rows


def _load_model_rows(model_name: str) -> list[dict[str, Any]]:
    path = output_root() / model_name / "boxes" / "predictions_conf020.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary() -> None:
    manifest = load_manifest()
    yolo_rows = {item["image"]: item for item in _load_model_rows("yolo_26x_img768")}
    stable_rows = {item["image"]: item for item in _load_model_rows("stable_dino_r50_img768")}
    out = output_root() / "compare" / "summary_counts.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["roi", "group", "image", "yolo_boxes", "stable_dino_boxes"])
        for item in manifest:
            image_name = item["image"]
            writer.writerow([
                item["roi"], item["group"], image_name,
                len(yolo_rows.get(image_name, {}).get("boxes", [])),
                len(stable_rows.get(image_name, {}).get("boxes", [])),
            ])


def make_side_by_side() -> None:
    manifest = load_manifest()
    compare_root = output_root() / "compare" / "side_by_side"
    yolo_root = output_root() / "yolo_26x_img768" / "overlays" / "all"
    stable_root = output_root() / "stable_dino_r50_img768" / "overlays" / "all"
    for item in manifest:
        image_name = str(item["image"])
        original = Path(item["input_path"])
        yolo = yolo_root / image_name
        stable = stable_root / image_name
        if not original.is_file() or not yolo.is_file() or not stable.is_file():
            continue
        panels = [Image.open(path).convert("RGB") for path in (original, yolo, stable)]
        width, height = panels[0].size
        canvas = Image.new("RGB", (width * 3, height), (255, 255, 255))
        for idx, panel in enumerate(panels):
            if panel.size != (width, height):
                panel = panel.resize((width, height))
            canvas.paste(panel, (idx * width, 0))
        out = compare_root / f"roi{item['roi']}_{Path(image_name).stem}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out, quality=92)


def write_stabledino_command() -> Path:
    prepare_inputs()
    raw_root = output_root() / "stable_dino_r50_img768" / "raw"
    command = [
        "python -m object_detection.semi_training.stable_dino.infer",
        f"--checkpoint {json.dumps(str(stable_dino_checkpoint()))}",
        f"--source {json.dumps(str(output_root() / 'inputs' / 'images'))}",
        f"--output-dir {json.dumps(str(raw_root))}",
        "--names " + " ".join(STABLE_DINO_NAMES),
        "--device auto",
        "--",
        "--",
        f"--imgsz {IMG_SIZE}",
        f"--test-with-nms {STABLE_DINO_NMS}",
    ]
    path = output_root() / "stable_dino_r50_img768" / "logs" / "stabledino_command.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(command) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run selected ROI detection workflow")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    yolo_parser = sub.add_parser("yolo")
    yolo_parser.add_argument("--device", default="auto")
    sub.add_parser("stabledino-normalize")
    sub.add_parser("summary")
    sub.add_parser("side-by-side")
    sub.add_parser("stabledino-command")
    all_yolo_parser = sub.add_parser("all-yolo")
    all_yolo_parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    if args.command == "prepare":
        manifest = prepare_inputs()
        print(f"Prepared {len(manifest)} images -> {output_root() / 'inputs' / 'images'}")
    elif args.command == "yolo":
        run_yolo(device=args.device)
    elif args.command == "stabledino-normalize":
        rows = normalize_stabledino()
        print(f"Normalized StableDINO detections for {len(rows)} images")
    elif args.command == "summary":
        write_summary()
        print(f"Wrote {output_root() / 'compare' / 'summary_counts.csv'}")
    elif args.command == "side-by-side":
        make_side_by_side()
        print(f"Wrote side-by-side images under {output_root() / 'compare' / 'side_by_side'}")
    elif args.command == "stabledino-command":
        path = write_stabledino_command()
        print(f"Wrote StableDINO command -> {path}")
    elif args.command == "all-yolo":
        run_yolo(device=args.device)
        write_summary()
        write_stabledino_command()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
