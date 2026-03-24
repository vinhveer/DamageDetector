from __future__ import annotations

import argparse

from inference_api.cli_support import log_to_stderr, parse_label_list, parse_queries, parse_roi, print_json
from .client import get_dino_service


def _common_params(args: argparse.Namespace) -> dict:
    params = {
        "gdino_checkpoint": args.checkpoint,
        "gdino_config_id": args.config_id,
        "text_queries": parse_queries(args.queries),
        "box_threshold": float(args.box_threshold),
        "text_threshold": float(args.text_threshold),
        "max_dets": int(args.max_dets),
        "device": args.device,
        "output_dir": args.output_dir,
    }
    roi = parse_roi(args.roi)
    if roi is not None:
        params["roi_box"] = roi
    return params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dino", description="CLI for the DINO engine.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--checkpoint", required=True, help="GroundingDINO checkpoint path or HF model id.")
        subparser.add_argument("--config-id", default="auto", help="GroundingDINO config id or local folder.")
        subparser.add_argument("--queries", default="crack", help="Comma-separated text queries.")
        subparser.add_argument("--box-threshold", type=float, default=0.25)
        subparser.add_argument("--text-threshold", type=float, default=0.25)
        subparser.add_argument("--max-dets", type=int, default=20)
        subparser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
        subparser.add_argument("--output-dir", default="results_dino")
        subparser.add_argument("--roi", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"))

    warmup = sub.add_parser("warmup", help="Warm up the DINO engine.")
    add_common(warmup)

    predict = sub.add_parser("predict", help="Run DINO detect-only on one image.")
    add_common(predict)
    predict.add_argument("--image", required=True)

    predict_batch = sub.add_parser("predict-batch", help="Run DINO detect-only on multiple images.")
    add_common(predict_batch)
    predict_batch.add_argument("--images", nargs="+", required=True)

    recursive = sub.add_parser("recursive-detect", help="Run recursive DINO detect-only for tiled crack search.")
    add_common(recursive)
    recursive.add_argument("--image", required=True)
    recursive.add_argument("--target-label", action="append", dest="target_labels", help="Label filter; repeat or use comma-separated values.")
    recursive.add_argument("--max-depth", type=int, default=3)
    recursive.add_argument("--min-box-px", type=int, default=48)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    service = get_dino_service()
    try:
        params = _common_params(args)
        if args.command == "warmup":
            result = service.call("warmup", {"params": params}, log_fn=log_to_stderr)
        elif args.command == "predict":
            result = service.call("predict", {"image_path": args.image, "params": params}, log_fn=log_to_stderr)
        elif args.command == "predict-batch":
            result = service.call("predict_batch", {"image_paths": list(args.images), "params": params}, log_fn=log_to_stderr)
        elif args.command == "recursive-detect":
            result = service.call(
                "recursive_detect",
                {
                    "image_path": args.image,
                    "params": params,
                    "target_labels": parse_label_list(args.target_labels),
                    "max_depth": int(args.max_depth),
                    "min_box_px": int(args.min_box_px),
                },
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
