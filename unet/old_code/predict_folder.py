import os
import argparse
import torch

from unet.unet_model import UNet
from predict import predict_image


def _iter_images(input_dir, recursive, exts):
    if recursive:
        for root, _, files in os.walk(input_dir):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext in exts:
                    yield os.path.join(root, name)
        return

    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in exts:
            yield path


def _safe_basename(input_dir, image_path):
    rel = os.path.relpath(image_path, input_dir)
    rel_no_ext = os.path.splitext(rel)[0]
    safe = rel_no_ext.replace("\\", "__").replace("/", "__")
    safe = safe.replace(":", "_")
    return safe


def _find_gt_mask(gt_dir, input_dir, image_path):
    if not gt_dir:
        return None

    rel = os.path.relpath(image_path, input_dir)
    rel_no_ext = os.path.splitext(rel)[0]
    candidate_roots = [
        os.path.join(gt_dir, rel_no_ext),
        os.path.join(gt_dir, os.path.splitext(os.path.basename(image_path))[0]),
    ]

    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]
    suffixes = ["", "_mask", "_gt", "_label", "_labels", "_seg", "_segmentation"]

    for root in candidate_roots:
        for suffix in suffixes:
            for ext in exts:
                cand = f"{root}{suffix}{ext}"
                if os.path.exists(cand):
                    return cand
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run crack prediction for every image in a folder."
    )
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Input folder containing images."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="output_results/best_model.pth",
        help="Model weights path.",
    )
    parser.add_argument("--output", type=str, default="results", help="Output directory.")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Binarization threshold in [0, 1]."
    )
    parser.add_argument(
        "--no-postprocessing", action="store_true", help="Disable post-processing."
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Scan subfolders recursively."
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default=None,
        help="Optional folder containing ground-truth masks (matched by filename / relative path).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tile", "letterbox", "resize"],
        default="tile",
        help="tile: sliding-window over original image; letterbox: keep aspect ratio; resize: stretch to square.",
    )
    parser.add_argument("--input-size", type=int, default=256, help="Model input size (square).")
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=0,
        help="Overlap (pixels) between tiles in tile mode. 0 means input_size//2 (recommended).",
    )
    parser.add_argument("--tile-batch-size", type=int, default=4, help="Batch size for tiles in tile mode.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: input folder does not exist: {args.input_dir}")
        return

    if not os.path.exists(args.model):
        print(f"Error: model file does not exist: {args.model}")
        return

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_paths = sorted(_iter_images(args.input_dir, args.recursive, exts))
    if not image_paths:
        print(f"No images found in: {args.input_dir}")
        return

    device = torch.device("cpu")
    print(f"Using device: {device}")

    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    print(f"Loaded model: {args.model}")
    print(f"Found {len(image_paths)} image(s).")

    for idx, image_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] {image_path}")
        output_basename = _safe_basename(args.input_dir, image_path)
        gt_mask_path = _find_gt_mask(args.gt_dir, args.input_dir, image_path)
        if args.gt_dir and not gt_mask_path:
            print(f"  [WARN] GT mask not found for: {image_path}")
        predict_image(
            model,
            image_path,
            device,
            threshold=args.threshold,
            output_dir=args.output,
            apply_postprocessing=not args.no_postprocessing,
            output_basename=output_basename,
            gt_mask_path=gt_mask_path,
            gt_expected=bool(args.gt_dir),
            mode=args.mode,
            input_size=args.input_size,
            tile_overlap=(args.input_size // 2) if args.tile_overlap == 0 else args.tile_overlap,
            tile_batch_size=args.tile_batch_size,
        )


if __name__ == "__main__":
    main()
