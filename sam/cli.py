from __future__ import annotations

import argparse

from inference_api.cli_support import load_boxes_json, log_to_stderr, parse_roi, print_json
from .client import get_sam_service


def _common_params(args: argparse.Namespace) -> dict:
    params = {
        "sam_checkpoint": args.checkpoint,
        "sam_model_type": args.sam_model_type,
        "invert_mask": bool(args.invert_mask),
        "sam_min_component_area": int(args.min_area),
        "sam_dilate_iters": int(args.dilate),
        "device": args.device,
        "output_dir": args.output_dir,
    }
    roi = parse_roi(args.roi)
    if roi is not None:
        params["roi_box"] = roi
    return params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sam", description="CLI for the SAM engine.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--checkpoint", required=True, help="SAM checkpoint path.")
        subparser.add_argument("--sam-model-type", default="auto", choices=["auto", "vit_b", "vit_l", "vit_h"])
        subparser.add_argument("--invert-mask", action="store_true")
        subparser.add_argument("--min-area", type=int, default=0)
        subparser.add_argument("--dilate", type=int, default=0)
        subparser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
        subparser.add_argument("--output-dir", default="results_sam")
        subparser.add_argument("--roi", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"))

    warmup = sub.add_parser("warmup", help="Warm up the SAM engine.")
    add_common(warmup)

    predict = sub.add_parser("predict", help="Run SAM-only prediction on one image.")
    add_common(predict)
    predict.add_argument("--image", required=True)

    predict_batch = sub.add_parser("predict-batch", help="Run SAM-only prediction on multiple images.")
    add_common(predict_batch)
    predict_batch.add_argument("--images", nargs="+", required=True)

    segment_boxes = sub.add_parser("segment-boxes", help="Run SAM box prompting from a boxes JSON file.")
    add_common(segment_boxes)
    segment_boxes.add_argument("--image", required=True)
    segment_boxes.add_argument("--boxes-json", required=True, help="JSON file containing detections/boxes.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    service = get_sam_service()
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
