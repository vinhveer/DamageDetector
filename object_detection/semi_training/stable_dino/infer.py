from __future__ import annotations

import argparse
import json
from pathlib import Path

from object_detection.semi_training.stable_dino.val import main as val_main


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run StableDINO inference on an image folder via a temporary COCO manifest.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--names", nargs="+", default=["crack", "mold", "spall"])
    parser.add_argument("--prototype-dir", default="")
    parser.add_argument("--dinov2-checkpoint", default="")
    parser.add_argument("--semantic-threshold", type=float, default=0.75)
    parser.add_argument("--semantic-reject-threshold", type=float, default=0.50)
    parser.add_argument("--semantic-margin-threshold", type=float, default=0.05)
    parser.add_argument("--semantic-mode", default="coverage", choices=["coverage", "class-consistency"])
    parser.add_argument("--semantic-batch-size", type=int, default=16)
    parser.add_argument("--expand-ratio", type=float, default=0.05)
    parser.add_argument("--prototype-cache", default="")
    parser.add_argument("--preview-limit", type=int, default=200)
    parser.add_argument("--previews", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("tail", nargs=argparse.REMAINDER)
    return parser


def _write_image_only_coco(source: Path, out_dir: Path, names: list[str]) -> Path:
    from PIL import Image

    source_is_file = source.is_file()
    image_paths = [source] if source_is_file else sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    root = out_dir / "image_only_coco"
    ann_dir = root / "annotations"
    image_dir = root / "images" / "val"
    ann_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    images = []
    seen_names: set[str] = set()
    for idx, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        base_name = image_path.name
        if base_name in seen_names:
            stem = image_path.stem
            suffix = image_path.suffix
            base_name = f"{stem}_{idx:06d}{suffix}"
        seen_names.add(base_name)
        link_path = image_dir / base_name
        if not link_path.exists():
            try:
                link_path.symlink_to(image_path.resolve())
            except OSError:
                import shutil

                shutil.copy2(image_path, link_path)
        images.append({"id": idx, "file_name": base_name, "width": width, "height": height})
    payload = {
        "images": images,
        "annotations": [],
        "categories": [{"id": idx, "name": name, "supercategory": "damage"} for idx, name in enumerate(names, start=1)],
    }
    ann_path = ann_dir / "instances_val.json"
    ann_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return root


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.output_dir).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()
    coco_root = _write_image_only_coco(source, out_dir, list(args.names))
    tail = [
        "--checkpoint", args.checkpoint,
        "--coco-root", str(coco_root),
        "--output-dir", str(out_dir),
        "--split", "val",
        "--device", args.device,
        "--semantic-threshold", str(args.semantic_threshold),
        "--semantic-reject-threshold", str(args.semantic_reject_threshold),
        "--semantic-margin-threshold", str(args.semantic_margin_threshold),
        "--semantic-mode", str(args.semantic_mode),
        "--semantic-batch-size", str(args.semantic_batch_size),
        "--expand-ratio", str(args.expand_ratio),
        "--preview-limit", str(args.preview_limit),
    ]
    if not args.previews:
        tail.append("--no-previews")
    if args.prototype_cache:
        tail.extend(["--prototype-cache", args.prototype_cache])
    if args.prototype_dir:
        tail.extend(["--prototype-dir", args.prototype_dir])
    if args.dinov2_checkpoint:
        tail.extend(["--dinov2-checkpoint", args.dinov2_checkpoint])
    if args.tail:
        tail.extend(args.tail[1:] if args.tail[0] == "--" else args.tail)
    return val_main(tail)


if __name__ == "__main__":
    raise SystemExit(main())
