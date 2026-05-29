"""Tiled YOLO inference with sliding window + NMS merge."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _nms(boxes: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x["score"], reverse=True)
    kept = []
    suppressed = [False] * len(boxes)
    for i, box in enumerate(boxes):
        if suppressed[i]:
            continue
        kept.append(box)
        for j in range(i + 1, len(boxes)):
            if not suppressed[j] and boxes[j]["label"] == box["label"]:
                if _iou(box["xyxy"], boxes[j]["xyxy"]) > iou_threshold:
                    suppressed[j] = True
    return kept


def tiled_predict(
    model_path: str,
    image_path: str,
    output_dir: str,
    tile_size: int = 512,
    overlap: int = 64,
    conf: float = 0.05,
    iou_nms: float = 0.5,
    device: str = "auto",
) -> list[dict]:
    from object_detection.yolo.lib import load_yolo_class, resolve_device

    YOLO = load_yolo_class()
    model = YOLO(str(Path(model_path).expanduser().resolve()))
    resolved_device = resolve_device(device)

    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    stride = tile_size - overlap

    all_boxes: list[dict] = []
    tiles_x = max(1, int(np.ceil((W - overlap) / stride)))
    tiles_y = max(1, int(np.ceil((H - overlap) / stride)))
    total = tiles_x * tiles_y
    print(f"Image {W}×{H} → {tiles_x}×{tiles_y} = {total} tiles ({tile_size}px, overlap={overlap})")

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            x1 = min(tx * stride, W - tile_size)
            y1 = min(ty * stride, H - tile_size)
            x2 = x1 + tile_size
            y2 = y1 + tile_size
            tile = image.crop((x1, y1, x2, y2))
            results = model.predict(
                source=tile,
                imgsz=tile_size,
                conf=conf,
                device=resolved_device,
                verbose=False,
            )
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                names = getattr(result, "names", {}) or {}
                xyxy = getattr(boxes, "xyxy", None)
                scores = getattr(boxes, "conf", None)
                classes = getattr(boxes, "cls", None)
                if xyxy is None:
                    continue
                for box, score, cls in zip(
                    xyxy.detach().cpu().tolist(),
                    scores.detach().cpu().tolist(),
                    classes.detach().cpu().tolist(),
                ):
                    all_boxes.append({
                        "label": str(names.get(int(cls), int(cls))),
                        "score": float(score),
                        "xyxy": [
                            box[0] + x1, box[1] + y1,
                            box[2] + x1, box[3] + y1,
                        ],
                    })
        print(f"  row {ty + 1}/{tiles_y} done, boxes so far: {len(all_boxes)}")

    print(f"Before NMS: {len(all_boxes)} boxes")
    kept = _nms(all_boxes, iou_threshold=iou_nms)
    print(f"After NMS:  {len(kept)} boxes")

    # Draw results — scale line width and font size to image resolution
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    colors = {"crack": (33, 150, 243), "spall": (0, 188, 212), "mold": (76, 175, 80)}
    line_w = max(4, W // 300)
    font_size = max(20, W // 80)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except Exception:
        font = ImageFont.load_default()
    for box in kept:
        color = colors.get(box["label"], (255, 87, 34))
        x1b, y1b, x2b, y2b = box["xyxy"]
        draw.rectangle([x1b, y1b, x2b, y2b], outline=color, width=line_w)
        label_text = f"{box['label']} {box['score']:.2f}"
        tw = font_size * len(label_text) * 0.6
        draw.rectangle([x1b, y1b - font_size - 6, x1b + tw, y1b], fill=color)
        draw.text((x1b + 4, y1b - font_size - 2), label_text, fill=(255, 255, 255), font=font)

    stem = Path(image_path).stem
    out_path = out_dir / f"{stem}_tiled.jpg"
    draw_image.save(out_path, quality=92)
    print(f"Saved: {out_path}")
    return kept


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tiled YOLO inference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou-nms", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    boxes = tiled_predict(
        model_path=args.model,
        image_path=args.source,
        output_dir=args.output_dir,
        tile_size=int(args.tile_size),
        overlap=int(args.overlap),
        conf=float(args.conf),
        iou_nms=float(args.iou_nms),
        device=args.device,
    )
    from inference_api.cli_support import print_json
    print_json({"total": len(boxes), "boxes": boxes}, pretty=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
