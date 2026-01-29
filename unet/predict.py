import argparse
import os
import torch

from unet.unet_model import UNet
from predict_lib.core import predict_image
from predict_lib.folder import predict_folder


def _build_parser():
    parser = argparse.ArgumentParser(description="Run crack prediction using a trained model.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Input image path.")
    group.add_argument("--input-dir", type=str, help="Input folder containing images.")
    parser.add_argument("--model", type=str, default="output_results/best_model.pth", help="Model weights path.")
    parser.add_argument("--output", type=str, default="results", help="Output directory.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold in [0, 1].")
    parser.add_argument("--no-postprocessing", action="store_true", help="Disable post-processing.")
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
    parser.add_argument("--recursive", action="store_true", help="Scan subfolders recursively (folder mode).")
    parser.add_argument(
        "--gt-dir",
        type=str,
        default=None,
        help="Optional folder containing ground-truth masks (matched by filename / relative path).",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: image file does not exist: {args.image}")
            return
    if args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"Error: input folder does not exist: {args.input_dir}")
            return
    if not os.path.exists(args.model):
        print(f"Error: model file does not exist: {args.model}")
        return

    device = torch.device("cpu")
    print(f"Using device: {device}")

    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    print(f"Loaded model: {args.model}")

    overlap = (args.input_size // 2) if args.tile_overlap == 0 else args.tile_overlap

    if args.image:
        predict_image(
            model,
            args.image,
            device,
            threshold=args.threshold,
            output_dir=args.output,
            apply_postprocessing=not args.no_postprocessing,
            mode=args.mode,
            input_size=args.input_size,
            tile_overlap=overlap,
            tile_batch_size=args.tile_batch_size,
        )
        return

    predict_folder(
        model,
        args.input_dir,
        device,
        output_dir=args.output,
        threshold=args.threshold,
        apply_postprocessing=not args.no_postprocessing,
        recursive=args.recursive,
        gt_dir=args.gt_dir,
        mode=args.mode,
        input_size=args.input_size,
        tile_overlap=args.tile_overlap,
        tile_batch_size=args.tile_batch_size,
    )


if __name__ == "__main__":
    main()
