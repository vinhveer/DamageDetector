from __future__ import annotations

import argparse
from pathlib import Path

from inference_api.cli_support import log_to_stderr, parse_label_list, parse_queries, parse_roi, print_json
from .engine import default_gdino_checkpoint
from .dinov2_classifier import default_dinov2_checkpoint
from .dinov2_prototypes import default_dinov2_embedding_checkpoint
from .prototype_dataset import build_prototypes_from_yolo_dataset
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
    if hasattr(args, "top_k"):
        params["top_k"] = int(args.top_k)
    if hasattr(args, "crop_dirname"):
        params["crop_dirname"] = args.crop_dirname
    if hasattr(args, "dinov2_checkpoint"):
        params["dinov2_checkpoint"] = args.dinov2_checkpoint
    if hasattr(args, "classifier_batch_size"):
        params["classifier_batch_size"] = int(args.classifier_batch_size)
    if hasattr(args, "classifier_top_k_labels"):
        params["classifier_top_k_labels"] = int(args.classifier_top_k_labels)
    if hasattr(args, "classifier_min_confidence"):
        params["classifier_min_confidence"] = float(args.classifier_min_confidence)
    if hasattr(args, "classifier_map_path"):
        params["classifier_map_path"] = args.classifier_map_path
    if hasattr(args, "classifier_strict"):
        params["classifier_strict"] = bool(args.classifier_strict)
    if hasattr(args, "prototype_dir"):
        params["prototype_dir"] = args.prototype_dir
    if hasattr(args, "prototype_batch_size"):
        params["prototype_batch_size"] = int(args.prototype_batch_size)
    if hasattr(args, "prototype_top_k_labels"):
        params["prototype_top_k_labels"] = int(args.prototype_top_k_labels)
    if hasattr(args, "prototype_min_similarity"):
        params["prototype_min_similarity"] = float(args.prototype_min_similarity)
    if hasattr(args, "prototype_background_labels"):
        params["prototype_background_labels"] = parse_label_list(args.prototype_background_labels)
    if hasattr(args, "prototype_strict"):
        params["prototype_strict"] = bool(args.prototype_strict)
    if hasattr(args, "save_overlay"):
        params["save_overlay"] = bool(args.save_overlay)
    if hasattr(args, "overlay_filename"):
        params["overlay_filename"] = str(args.overlay_filename or "").strip()
    if hasattr(args, "overlay_include_rejected"):
        params["overlay_include_rejected"] = bool(args.overlay_include_rejected)
    if hasattr(args, "mm_per_px"):
        params["mm_per_px"] = float(args.mm_per_px)
    roi = parse_roi(args.roi)
    if roi is not None:
        params["roi_box"] = roi
    return params


def _checkpoint_help(default_path: str) -> str:
    if default_path and Path(default_path).exists():
        return f"GroundingDINO checkpoint path or HF model id. Local default: {default_path}"
    return "GroundingDINO checkpoint path or HF model id."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dino", description="CLI for the DINO engine.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        default_checkpoint = default_gdino_checkpoint()
        has_default_checkpoint = bool(default_checkpoint and Path(default_checkpoint).exists())
        subparser.add_argument(
            "--checkpoint",
            required=not has_default_checkpoint,
            default=default_checkpoint if has_default_checkpoint else None,
            help=_checkpoint_help(default_checkpoint),
        )
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

    build_proto = sub.add_parser("build-prototypes", help="Build few-shot prototype crops from a YOLO dataset export.")
    build_proto.add_argument("--dataset-dir", required=True, help="YOLO dataset root containing data.yaml and split folders.")
    build_proto.add_argument("--output-dir", required=True, help="Where to save prototype class folders.")
    build_proto.add_argument("--split", action="append", dest="splits", help="Dataset split to include; repeat or use comma-separated values.")
    build_proto.add_argument("--samples-per-class", type=int, default=24, help="Maximum positive crops to save per class.")
    build_proto.add_argument("--background-samples", type=int, default=24, help="How many background crops to generate.")
    build_proto.add_argument("--pad-ratio", type=float, default=0.08, help="Extra padding around each GT box before cropping.")
    build_proto.add_argument("--min-crop-size", type=int, default=64, help="Skip crops smaller than this many pixels on either side.")
    build_proto.add_argument("--seed", type=int, default=7, help="Random seed for crop sampling.")

    rank = sub.add_parser("rank-boxes", help="Tile image like editor DINO, relabel proposal crops with DINOv2, then keep top-k.")
    add_common(rank)
    rank.add_argument("--image", required=True)
    rank.add_argument("--top-k", type=int, default=1, help="How many highest-score DINO boxes to keep.")
    rank.add_argument("--crop-dirname", default="dino_crops", help="Subfolder inside output_dir where crops are saved.")
    rank.add_argument(
        "--dinov2-checkpoint",
        default="",
        help=(
            "Optional DINOv2 checkpoint/model folder. Defaults to "
            f"{default_dinov2_embedding_checkpoint()} in prototype mode, or {default_dinov2_checkpoint()} in classifier mode."
        ),
    )
    rank.add_argument("--classifier-batch-size", type=int, default=8, help="DINOv2 classifier batch size.")
    rank.add_argument("--classifier-top-k-labels", type=int, default=3, help="How many top classifier labels to inspect per crop.")
    rank.add_argument("--classifier-min-confidence", type=float, default=0.0, help="Reject proposal when classifier top-1 confidence is below this value.")
    rank.add_argument("--classifier-map-path", default="", help="Path to JSON mapping rules, or inline JSON string.")
    rank.add_argument("--classifier-strict", action="store_true", help="Reject crops that the classifier cannot map to a target label.")
    rank.add_argument("--prototype-dir", default="", help="Few-shot support set root. Expected layout: prototype_dir/<label>/*.png")
    rank.add_argument("--prototype-batch-size", type=int, default=8, help="DINOv2 embedding batch size for prototype mode.")
    rank.add_argument("--prototype-top-k-labels", type=int, default=3, help="How many nearest prototype labels to keep per crop.")
    rank.add_argument("--prototype-min-similarity", type=float, default=0.3, help="Reject proposal when best prototype cosine similarity is below this value.")
    rank.add_argument(
        "--prototype-background-label",
        action="append",
        dest="prototype_background_labels",
        help="Prototype label that should reject a proposal; repeat or use comma-separated values.",
    )
    rank.add_argument(
        "--prototype-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In prototype mode, reject crops with no confident non-background prototype match.",
    )
    rank.add_argument("--save-overlay", action="store_true", help="Write overlay image with filtered boxes to output_dir.")
    rank.add_argument("--overlay-filename", default="overlay_filtered.png", help="Filename for overlay image inside output_dir.")
    rank.add_argument("--overlay-include-rejected", action="store_true", help="Also draw rejected boxes in blue.")
    rank.add_argument(
        "--mm-per-px",
        type=float,
        default=2.11,
        help="Physical scale for area overlay. If > 0, show box area using 1px = N mm (default: 2.11).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "build-prototypes":
        result = build_prototypes_from_yolo_dataset(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            splits=parse_label_list(args.splits) or ["train", "valid"],
            samples_per_class=int(args.samples_per_class),
            background_samples=int(args.background_samples),
            pad_ratio=float(args.pad_ratio),
            min_crop_size=int(args.min_crop_size),
            seed=int(args.seed),
        )
        print_json(result, pretty=bool(args.pretty))
        return 0

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
        elif args.command == "rank-boxes":
            result = service.call(
                "rank_boxes",
                {
                    "image_path": args.image,
                    "params": params,
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
