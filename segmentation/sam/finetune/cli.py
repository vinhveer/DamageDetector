from __future__ import annotations

import argparse

from inference_api.cli_support import load_boxes_json, log_to_stderr, parse_roi, print_json
from .client import get_sam_finetune_service


def _common_params(args: argparse.Namespace) -> dict:
    params = {
        "sam_checkpoint": args.checkpoint,
        "sam_model_type": args.sam_model_type,
        "delta_type": args.delta_type,
        "delta_checkpoint": args.delta_checkpoint,
        "centerline_head": bool(args.centerline_head),
        "middle_dim": int(args.middle_dim),
        "scaling_factor": float(args.scaling_factor),
        "rank": int(args.rank),
        "invert_mask": bool(args.invert_mask),
        "sam_min_component_area": int(args.min_area),
        "sam_dilate_iters": int(args.dilate),
        "device": args.device,
        "output_dir": args.output_dir,
        "predict_mode": args.predict_mode,
        "tile_size": int(args.tile_size),
        "tile_overlap": int(args.tile_overlap),
        "tile_batch_size": int(args.tile_batch_size),
        "threshold": args.threshold,
        "refine_delta_checkpoint": args.refine_delta_checkpoint,
        "refine_delta_type": args.refine_delta_type,
        "refine_rank": int(args.refine_rank),
        "refine_decoder_type": args.refine_decoder_type,
        "refine_centerline_head": bool(args.refine_centerline_head),
        "refine_tile_size": int(args.refine_tile_size),
        "refine_tile_sizes": tuple(int(v) for v in (args.refine_tile_sizes or []) if int(v) > 0),
        "refine_batch_size": int(args.refine_batch_size),
        "refine_max_rois": int(args.refine_max_rois),
        "refine_roi_padding": int(args.refine_roi_padding),
        "refine_merge_mode": args.refine_merge_mode,
        "refine_score_threshold": float(args.refine_score_threshold),
        "refine_positive_band_low": float(args.refine_positive_band_low),
        "refine_positive_band_high": float(args.refine_positive_band_high),
    }
    roi = parse_roi(args.roi)
    if roi is not None:
        params["roi_box"] = roi
    return params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m segmentation.sam.finetune", description="CLI for the SAM finetune engine.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--checkpoint", required=True, help="Base SAM checkpoint path.")
        subparser.add_argument("--sam-model-type", default="auto", choices=["auto", "vit_b", "vit_l", "vit_h"])
        subparser.add_argument("--delta-type", required=True, choices=["adapter", "lora", "both"])
        subparser.add_argument("--delta-checkpoint", default="auto")
        subparser.add_argument("--centerline-head", action="store_true")
        subparser.add_argument("--middle-dim", type=int, default=32)
        subparser.add_argument("--scaling-factor", type=float, default=0.2)
        subparser.add_argument("--rank", type=int, default=4)
        subparser.add_argument("--invert-mask", action="store_true")
        subparser.add_argument("--min-area", type=int, default=0)
        subparser.add_argument("--dilate", type=int, default=0)
        subparser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
        subparser.add_argument("--output-dir", default="results_sam_finetune")
        subparser.add_argument("--predict-mode", default="auto", choices=["auto", "tile_full_box", "legacy_full_box", "coarse_refine"])
        subparser.add_argument("--tile-size", type=int, default=-1, help="Tile size for tile_full_box mode (-1 = use checkpoint metadata or 512).")
        subparser.add_argument("--tile-overlap", type=int, default=-1, help="Tile overlap for tile_full_box mode (-1 = use checkpoint metadata or tile_size // 2).")
        subparser.add_argument("--tile-batch-size", type=int, default=4, help="Batch size for tiled full-image inference.")
        subparser.add_argument("--threshold", default="auto", help="Mask threshold as float or 'auto' to use best_threshold.txt.")
        subparser.add_argument("--refine-delta-checkpoint", default="")
        subparser.add_argument("--refine-delta-type", default="")
        subparser.add_argument("--refine-rank", type=int, default=-1)
        subparser.add_argument("--refine-decoder-type", default="auto", choices=["auto", "baseline", "hq"])
        subparser.add_argument("--refine-centerline-head", action="store_true")
        subparser.add_argument("--refine-tile-size", type=int, default=-1)
        subparser.add_argument("--refine-tile-sizes", type=int, nargs="*", default=None)
        subparser.add_argument("--refine-batch-size", type=int, default=2, help="Batch size for refine ROI inference.")
        subparser.add_argument("--refine-max-rois", type=int, default=16)
        subparser.add_argument("--refine-roi-padding", type=int, default=64)
        subparser.add_argument("--refine-merge-mode", default="weighted_replace")
        subparser.add_argument("--refine-score-threshold", type=float, default=0.15)
        subparser.add_argument("--refine-positive-band-low", type=float, default=0.20)
        subparser.add_argument("--refine-positive-band-high", type=float, default=0.90)
        subparser.add_argument("--roi", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"))

    warmup = sub.add_parser("warmup", help="Warm up the SAM finetune engine.")
    add_common(warmup)

    predict = sub.add_parser("predict", help="Run SAM finetune prediction on one image.")
    add_common(predict)
    predict.add_argument("--image", required=True)

    predict_batch = sub.add_parser("predict-batch", help="Run SAM finetune prediction on multiple images.")
    add_common(predict_batch)
    predict_batch.add_argument("--images", nargs="+", required=True)

    segment_boxes = sub.add_parser("segment-boxes", help="Run SAM finetune box prompting from a boxes JSON file.")
    add_common(segment_boxes)
    segment_boxes.add_argument("--image", required=True)
    segment_boxes.add_argument("--boxes-json", required=True, help="JSON file containing detections/boxes.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    service = get_sam_finetune_service()
    try:
        params = _common_params(args)
        if args.command == "warmup":
            result = service.call("warmup", {"params": params}, log_fn=log_to_stderr)
        elif args.command == "predict":
            result = service.call("predict", {"image_path": args.image, "params": params}, log_fn=log_to_stderr)
        elif args.command == "predict-batch":
            result = service.call("predict_batch", {"image_paths": list(args.images), "params": params}, log_fn=log_to_stderr)
        elif args.command == "segment-boxes":
            result = service.call(
                "segment_boxes",
                {"image_path": args.image, "params": params, "boxes": load_boxes_json(args.boxes_json)},
                log_fn=log_to_stderr,
            )
        else:
            raise ValueError(f"Unknown command: {args.command}")
        print_json(result, pretty=bool(args.pretty))
        return 0
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
