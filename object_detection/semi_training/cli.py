from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inference_api.cli_support import print_json

from .common.coco import list_split_image_paths, list_split_image_paths_by_class
from .common.predictions import coco_annotations_as_predictions, load_predictions_json
from .common.prototype_bank import build_prototype_crop_bank
from .common.semantic_validation import semantic_validate_predictions
from .common.stable_format import write_stable_dino_dataset_yaml
from .common.yolo_format import export_coco_root_to_yolo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m object_detection.semi_training")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-dataset", help="Prepare YOLO and StableDINO configs from a semi-label COCO dataset")
    prepare.add_argument("--coco-root", required=True)
    prepare.add_argument("--output-dir", default=None, help="YOLO output dir. Default: <coco-root>/yolo")
    prepare.add_argument("--link-mode", default="symlink", choices=["symlink", "copy"])
    prepare.add_argument("--stable-yaml", default=None)
    prepare.add_argument("--pretty", action="store_true")

    prepare_stable = sub.add_parser("prepare-stable-dino", help="Prepare only the StableDINO dataset YAML from a semi-label COCO dataset")
    prepare_stable.add_argument("--coco-root", required=True)
    prepare_stable.add_argument("--output", default=None, help="Output YAML path. Default: <coco-root>/stable_dino_dataset.yaml")
    prepare_stable.add_argument("--pretty", action="store_true")

    proto = sub.add_parser("build-prototypes", help="Build prototype crop folders from COCO annotations")
    proto.add_argument("--coco-root", required=True)
    proto.add_argument("--output-dir", required=True)
    proto.add_argument("--splits", nargs="+", default=["train"])
    proto.add_argument("--samples-per-class", type=int, default=32)
    proto.add_argument("--min-crop-size", type=int, default=32)
    proto.add_argument("--seed", type=int, default=7)
    proto.add_argument("--pretty", action="store_true")

    validate = sub.add_parser("semantic-val", help="Validate prediction crops against DINOv2 prototypes")
    validate.add_argument("--predictions-json", default="", help="Prediction JSON from detector. If omitted, use COCO annotations as pseudo-predictions.")
    validate.add_argument("--coco-root", default="")
    validate.add_argument("--split", default="val")
    validate.add_argument("--limit", type=int, default=0)
    validate.add_argument("--output-dir", required=True)
    validate.add_argument("--prototype-dir", default="")
    validate.add_argument("--dinov2-checkpoint", default="")
    validate.add_argument("--device", default="auto")
    validate.add_argument("--batch-size", type=int, default=16)
    validate.add_argument("--threshold", type=float, default=0.75)
    validate.add_argument("--reject-threshold", type=float, default=0.50)
    validate.add_argument("--margin-threshold", type=float, default=0.05)
    validate.add_argument("--decision-mode", default="coverage", choices=["coverage", "class-consistency"])
    validate.add_argument("--expand-ratio", type=float, default=0.05)
    validate.add_argument("--prototype-cache", default="")
    validate.add_argument("--preview-limit", type=int, default=200)
    validate.add_argument("--previews", action=argparse.BooleanOptionalAction, default=True)
    validate.add_argument("--pretty", action="store_true")

    yolo = sub.add_parser("yolo", help="YOLO semi-training commands")
    yolo_sub = yolo.add_subparsers(dest="yolo_command", required=True)
    yolo_sub.add_parser("train")
    yolo_sub.add_parser("val")
    yolo_sub.add_parser("infer")

    stable = sub.add_parser("stable-dino", help="StableDINO semi-training commands")
    stable_sub = stable.add_subparsers(dest="stable_command", required=True)
    stable_sub.add_parser("train")
    stable_sub.add_parser("val")
    stable_sub.add_parser("infer")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == "yolo" and len(raw_argv) >= 2:
        command = raw_argv[1]
        tail = raw_argv[2:]
        if tail and tail[0] == "--":
            tail = tail[1:]
        if command == "train":
            from .yolo.train import main as command_main
        elif command == "val":
            from .yolo.val import main as command_main
        elif command == "infer":
            from .yolo.infer import main as command_main
        else:
            raise ValueError("YOLO command must be one of: train, val, infer")
        return command_main(tail)
    if raw_argv and raw_argv[0] == "stable-dino" and len(raw_argv) >= 2:
        command = raw_argv[1]
        tail = raw_argv[2:]
        if tail and tail[0] == "--":
            tail = tail[1:]
        if command == "train":
            from .stable_dino.train import main as command_main
        elif command == "val":
            from .stable_dino.val import main as command_main
        elif command == "infer":
            from .stable_dino.infer import main as command_main
        else:
            raise ValueError("StableDINO command must be one of: train, val, infer")
        return command_main(tail)

    parser = build_parser()
    args, tail = parser.parse_known_args(raw_argv)

    if args.command == "prepare-dataset":
        coco_root = Path(args.coco_root).expanduser().resolve()
        yolo = export_coco_root_to_yolo(coco_root=coco_root, output_dir=args.output_dir or coco_root / "yolo", link_mode=args.link_mode)
        stable = write_stable_dino_dataset_yaml(coco_root=coco_root, output_path=args.stable_yaml)
        print_json({"status": "ok", "yolo": yolo, "stable_dino": stable}, pretty=bool(args.pretty))
        return 0

    if args.command == "prepare-stable-dino":
        result = write_stable_dino_dataset_yaml(coco_root=args.coco_root, output_path=args.output)
        print_json({"status": "ok", "stable_dino": result}, pretty=bool(args.pretty))
        return 0

    if args.command == "build-prototypes":
        result = build_prototype_crop_bank(
            coco_root=args.coco_root,
            output_dir=args.output_dir,
            splits=tuple(args.splits),
            samples_per_class=int(args.samples_per_class),
            min_crop_size=int(args.min_crop_size),
            seed=int(args.seed),
        )
        print_json({"status": "ok", **result}, pretty=bool(args.pretty))
        return 0

    if args.command == "semantic-val":
        if args.predictions_json:
            predictions = load_predictions_json(args.predictions_json)
            expected_image_paths = list_split_image_paths(args.coco_root, args.split) if args.coco_root else None
            expected_per_class = list_split_image_paths_by_class(args.coco_root, args.split) if args.coco_root else None
        elif args.coco_root:
            predictions = coco_annotations_as_predictions(coco_root=args.coco_root, split_name=args.split, limit=int(args.limit))
            expected_image_paths = list_split_image_paths(args.coco_root, args.split)
            expected_per_class = list_split_image_paths_by_class(args.coco_root, args.split)
        else:
            raise ValueError("Either --predictions-json or --coco-root is required")
        result = semantic_validate_predictions(
            predictions=predictions,
            output_dir=args.output_dir,
            prototype_dir=args.prototype_dir or None,
            dinov2_checkpoint=args.dinov2_checkpoint or None,
            device=args.device,
            batch_size=int(args.batch_size),
            threshold=float(args.threshold),
            reject_threshold=float(args.reject_threshold),
            margin_threshold=float(args.margin_threshold),
            decision_mode=str(args.decision_mode),
            expand_ratio=float(args.expand_ratio),
            prototype_cache_path=args.prototype_cache or None,
            expected_image_paths=expected_image_paths,
            expected_per_class=expected_per_class,
            save_previews=bool(args.previews),
            preview_limit=int(args.preview_limit),
        )
        print_json({"status": "ok", **result}, pretty=bool(args.pretty))
        return 0

    if args.command == "yolo":
        if args.yolo_command == "train":
            from .yolo.train import main as command_main
        elif args.yolo_command == "val":
            from .yolo.val import main as command_main
        else:
            from .yolo.infer import main as command_main
        return command_main(tail)

    if args.command == "stable-dino":
        if args.stable_command == "train":
            from .stable_dino.train import main as command_main
        elif args.stable_command == "val":
            from .stable_dino.val import main as command_main
        else:
            from .stable_dino.infer import main as command_main
        return command_main(tail)

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
