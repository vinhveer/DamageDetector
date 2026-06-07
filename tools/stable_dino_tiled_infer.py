"""Tiled StableDINO inference with sliding window + NMS merge."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
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


def _load_model(checkpoint_dir: str, device: str):
    """Load StableDINO model from a training output directory using saved config.yaml.pkl."""
    import pickle
    import sys
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import instantiate

    # `projects` package lives inside object_detection/stable_dino/
    _stable_dino_root = Path(__file__).resolve().parent.parent / "object_detection" / "stable_dino"
    if str(_stable_dino_root) not in sys.path:
        sys.path.insert(0, str(_stable_dino_root))

    ckpt_dir = Path(checkpoint_dir).expanduser().resolve()
    pkl_path = ckpt_dir / "config.yaml.pkl"
    model_path = ckpt_dir / "model_best.pth"
    if not model_path.exists():
        model_path = ckpt_dir / "model_final.pth"

    with open(pkl_path, "rb") as f:
        cfg = pickle.load(f)

    from omegaconf import OmegaConf
    cfg.model.device = device
    model = instantiate(cfg.model)
    model.to(device)
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(str(model_path))
    print(f"Loaded: {model_path.name}")
    return model, cfg


def _predict_tile(model, tile_rgb: np.ndarray, device: str) -> list[dict]:
    """Run model on a single HxWx3 uint8 tile, return list of {label, score, xyxy}."""
    # detectron2 expects CHW float32 tensor
    img_tensor = torch.as_tensor(tile_rgb.transpose(2, 0, 1), dtype=torch.float32).to(device)
    h, w = tile_rgb.shape[:2]
    with torch.no_grad():
        outputs = model([{"image": img_tensor, "height": h, "width": w}])
    instances = outputs[0]["instances"].to("cpu")
    boxes_out = []
    for box, cls, score in zip(
        instances.pred_boxes.tensor.tolist(),
        instances.pred_classes.tolist(),
        instances.scores.tolist(),
    ):
        boxes_out.append({"label": int(cls), "score": float(score), "xyxy": box})
    return boxes_out


def tiled_predict(
    checkpoint_dir: str,
    image_path: str,
    output_dir: str,
    class_names: list[str],
    tile_size: int = 512,
    overlap: int = 64,
    conf: float = 0.05,
    iou_nms: float = 0.5,
    device: str = "cpu",
) -> list[dict]:
    model, _cfg = _load_model(checkpoint_dir, device)

    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    stride = tile_size - overlap

    tiles_x = max(1, int(np.ceil((W - overlap) / stride)))
    tiles_y = max(1, int(np.ceil((H - overlap) / stride)))
    total = tiles_x * tiles_y
    print(f"Image {W}×{H} → {tiles_x}×{tiles_y} = {total} tiles ({tile_size}px, overlap={overlap})")

    all_boxes: list[dict] = []
    img_arr = np.array(image)

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            x1 = min(tx * stride, W - tile_size)
            y1 = min(ty * stride, H - tile_size)
            x2 = x1 + tile_size
            y2 = y1 + tile_size
            tile_arr = img_arr[y1:y2, x1:x2]
            raw = _predict_tile(model, tile_arr, device)
            for b in raw:
                if b["score"] < conf:
                    continue
                bx1, by1, bx2, by2 = b["xyxy"]
                all_boxes.append({
                    "label": class_names[b["label"]] if b["label"] < len(class_names) else str(b["label"]),
                    "score": b["score"],
                    "xyxy": [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
                })
        print(f"  row {ty + 1}/{tiles_y} done, boxes so far: {len(all_boxes)}")

    print(f"Before NMS: {len(all_boxes)} boxes")
    kept = _nms(all_boxes, iou_threshold=iou_nms)
    print(f"After NMS:  {len(kept)} boxes")

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
    out_path = out_dir / f"{stem}_stabledino_tiled.jpg"
    draw_image.save(out_path, quality=92)
    print(f"Saved: {out_path}")
    return kept


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tiled StableDINO inference")
    parser.add_argument("--checkpoint-dir", required=True, help="Training output dir with config.yaml + model_best.pth")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--names", nargs="+", default=["crack", "mold", "spall"])
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou-nms", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    kept = tiled_predict(
        checkpoint_dir=args.checkpoint_dir,
        image_path=args.source,
        output_dir=args.output_dir,
        class_names=args.names,
        tile_size=args.tile_size,
        overlap=args.overlap,
        conf=args.conf,
        iou_nms=args.iou_nms,
        device=args.device,
    )
    import json
    print(json.dumps({"total": len(kept), "boxes": kept}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
