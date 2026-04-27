#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from clip_model import OpenClipSemanticClassifier


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test OpenCLIP semantic labeling on one image.")
    parser.add_argument("--image", required=True, help="Image path to classify.")
    parser.add_argument("--model-name", default="ViT-B-32", help="OpenCLIP model name.")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k", help="OpenCLIP pretrained tag.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output-json", default="", help="Optional JSON output path.")
    args = parser.parse_args(argv)

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    classifier = OpenClipSemanticClassifier(
        model_name=str(args.model_name),
        pretrained=str(args.pretrained),
        device=str(args.device),
    )
    with Image.open(image_path) as image:
        result = classifier.classify_image(image)
    result["image_path"] = str(image_path)

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if str(args.output_json or "").strip():
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
