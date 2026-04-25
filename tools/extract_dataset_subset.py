from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _resolve_mask_path(masks_dir: Path, image_path: Path) -> Path:
    direct = masks_dir / image_path.name
    if direct.is_file():
        return direct
    matches = sorted(path for path in masks_dir.glob(f"{image_path.stem}.*") if path.is_file())
    if not matches:
        raise FileNotFoundError(f"Missing mask for image: {image_path.name}")
    return matches[0]


def _iter_matches(images_dir: Path, token: str):
    needle = str(token).strip().lower()
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        if needle in image_path.name.lower():
            yield image_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract a subset dataset by filename token.")
    parser.add_argument("--input-root", required=True, type=Path, help="Dataset root containing images/ and masks/")
    parser.add_argument("--output-root", required=True, type=Path, help="Destination dataset root")
    parser.add_argument("--match-token", required=True, help="Case-insensitive filename token")
    parser.add_argument("--overwrite", action="store_true", help="Remove output root before writing")
    args = parser.parse_args()

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    images_dir = input_root / "images"
    masks_dir = input_root / "masks"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images folder: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Missing masks folder: {masks_dir}")

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)

    out_images = output_root / "images"
    out_masks = output_root / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    selected_images = list(_iter_matches(images_dir, args.match_token))
    if not selected_images:
        raise FileNotFoundError(f"No images matched token {args.match_token!r} in {images_dir}")

    rows: list[dict[str, str]] = []
    for image_path in selected_images:
        mask_path = _resolve_mask_path(masks_dir, image_path)
        shutil.copy2(image_path, out_images / image_path.name)
        shutil.copy2(mask_path, out_masks / mask_path.name)
        rows.append(
            {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "mask_name": mask_path.name,
                "mask_path": str(mask_path),
            }
        )

    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "match_token": str(args.match_token),
        "image_count": len(rows),
        "files": rows,
    }
    manifest_path = output_root / "subset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(str(output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
