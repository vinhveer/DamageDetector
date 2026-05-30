"""Render StableDINO detection overlays from saved COCO results.

The `damage-stable-dino-infer` command runs detectron2 eval-only and writes
`coco_instances_results.json` (predictions) + `dataset_cache/<split>_coco.json`
(image_id -> file_name mapping) but no visual overlays.

This tool draws those predictions onto the source images and saves them into a
`predict/` folder, mirroring the YOLO `inference/predict/` layout so the two
detectors can be compared side by side. It does NOT re-run the model.

Example
-------
    python -m tools.stable_dino_render_overlays \
        --inference-dir "/.../stable_dino_r50_img768/inference" \
        --conf 0.25
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# BGR-free RGB palette, keyed by class name (falls back to orange).
_COLORS = {
    "crack": (33, 150, 243),
    "spall": (0, 188, 212),
    "mold": (76, 175, 80),
}
_FALLBACK_COLOR = (255, 87, 34)


def _load_font(font_size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def _resolve_coco_meta(inference_dir: Path, eval_split: str) -> Path:
    cache = inference_dir / "dataset_cache" / f"{eval_split}_coco.json"
    if cache.exists():
        return cache
    # Fall back to whatever split json is present.
    candidates = sorted((inference_dir / "dataset_cache").glob("*_coco.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No COCO meta json found in {inference_dir / 'dataset_cache'}; "
            "expected e.g. test_coco.json"
        )
    return candidates[0]


def render_overlays(
    inference_dir: Path,
    eval_split: str = "test",
    conf: float = 0.25,
    output_name: str = "predict",
    max_images: int | None = None,
) -> Path:
    inference_dir = inference_dir.expanduser().resolve()
    results_path = inference_dir / "coco_instances_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing predictions: {results_path}")

    meta_path = _resolve_coco_meta(inference_dir, eval_split)
    coco = json.loads(meta_path.read_text())

    id_to_image = {img["id"]: img for img in coco["images"]}
    cat_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}

    predictions = json.loads(results_path.read_text())
    preds_by_image: dict[int, list[dict]] = defaultdict(list)
    for pred in predictions:
        if float(pred.get("score", 0.0)) < conf:
            continue
        preds_by_image[pred["image_id"]].append(pred)

    out_dir = inference_dir / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    image_ids = list(id_to_image.keys())
    if max_images is not None:
        image_ids = image_ids[:max_images]

    rendered = 0
    skipped_missing = 0
    for image_id in image_ids:
        info = id_to_image[image_id]
        src = Path(info["file_name"])
        if not src.exists():
            skipped_missing += 1
            continue

        image = Image.open(src).convert("RGB")
        W, _H = image.size
        draw = ImageDraw.Draw(image)
        line_w = max(2, W // 320)
        font_size = max(14, W // 45)
        font = _load_font(font_size)

        boxes = sorted(preds_by_image.get(image_id, []), key=lambda b: b["score"])
        for box in boxes:
            name = cat_to_name.get(box["category_id"], str(box["category_id"]))
            color = _COLORS.get(name, _FALLBACK_COLOR)
            x, y, w, h = box["bbox"]  # COCO xywh
            x2, y2 = x + w, y + h
            draw.rectangle([x, y, x2, y2], outline=color, width=line_w)
            label = f"{name} {box['score']:.2f}"
            tw = font_size * len(label) * 0.6
            draw.rectangle([x, max(0, y - font_size - 6), x + tw, y], fill=color)
            draw.text((x + 3, max(0, y - font_size - 4)), label, fill=(255, 255, 255), font=font)

        image.save(out_dir / src.name, quality=92)
        rendered += 1

    print(f"Rendered {rendered} overlays -> {out_dir}")
    if skipped_missing:
        print(f"Skipped {skipped_missing} images (source file not found)")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render StableDINO overlays from saved COCO results")
    parser.add_argument("--inference-dir", required=True, help="Stable DINO inference output dir")
    parser.add_argument("--eval-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--conf", type=float, default=0.25, help="Score threshold (match YOLO conf)")
    parser.add_argument("--name", default="predict", help="Output subfolder name")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images (debug)")
    args = parser.parse_args(argv)
    render_overlays(
        inference_dir=Path(args.inference_dir),
        eval_split=args.eval_split,
        conf=args.conf,
        output_name=args.name,
        max_images=args.max_images,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
