from __future__ import annotations

import argparse

from inference_api.cli_support import parse_roi, print_json, log_to_stderr
from .client import get_unet_service


def _common_params(args: argparse.Namespace) -> dict:
    params = {
        "model_path": args.model,
        "output_dir": args.output_dir,
        "threshold": float(args.threshold),
        "apply_postprocessing": not bool(args.no_postprocessing),
        "mode": args.mode,
        "input_size": int(args.input_size),
        "tile_overlap": int(args.tile_overlap),
        "tile_batch_size": int(args.tile_batch_size),
        "device": args.device,
    }
    roi = parse_roi(args.roi)
    if roi is not None:
        params["roi_box"] = roi
    return params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m segmentation.unet", description="CLI for the UNet engine.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--model", required=True, help="UNet checkpoint path.")
        subparser.add_argument("--output-dir", default="results_unet")
        subparser.add_argument("--threshold", type=float, default=0.5)
        subparser.add_argument("--no-postprocessing", action="store_true")
        subparser.add_argument("--mode", default="tile", choices=["tile", "letterbox", "resize"])
        subparser.add_argument("--input-size", type=int, default=512)
        subparser.add_argument("--tile-overlap", type=int, default=0)
        subparser.add_argument("--tile-batch-size", type=int, default=4)
        subparser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
        subparser.add_argument("--roi", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"))

    warmup = sub.add_parser("warmup", help="Warm up the UNet engine.")
    add_common(warmup)

    predict = sub.add_parser("predict", help="Run UNet on one image.")
    add_common(predict)
    predict.add_argument("--image", required=True)

    predict_batch = sub.add_parser("predict-batch", help="Run UNet on multiple images.")
    add_common(predict_batch)
    predict_batch.add_argument("--images", nargs="+", required=True)

    run_rois = sub.add_parser("run-rois", help="Run UNet on one image using one or more explicit ROIs.")
    add_common(run_rois)
    run_rois.add_argument("--image", required=True)
    run_rois.add_argument("--roi-box", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"), action="append", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    service = get_unet_service()
    try:
        params = _common_params(args)
        if args.command == "warmup":
            result = service.call("warmup", {"params": params}, log_fn=log_to_stderr)
        elif args.command == "predict":
            result = service.call("predict", {"image_path": args.image, "params": params}, log_fn=log_to_stderr)
        elif args.command == "predict-batch":
            result = service.call("predict_batch", {"image_paths": list(args.images), "params": params}, log_fn=log_to_stderr)
        elif args.command == "run-rois":
            rois = [parse_roi(list(values)) for values in args.roi_box]
            result = service.call(
                "run_rois",
                {"image_path": args.image, "params": params, "rois": [roi for roi in rois if roi is not None]},
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
