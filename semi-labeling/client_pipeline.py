#!/usr/bin/env python3
"""Client-facing semi-labeling pipeline.

This CLI hides the internal step01/02/... names and exposes a simple workflow:

    detect   input images -> GDINO candidate boxes
    filter   OpenCLIP semantic check + initial clean/review queues
    prepare  crops + DINOv2 embeddings + core clusters for review
    clean    optional seed/reliability/policy refresh after prototype/review
    export   cleaned labels -> YOLO/COCO dataset

Recommended use:

    python -m client_pipeline detect  --input-dir /content/HinhAnh --output-dir /content/out
    python -m client_pipeline filter  --input-dir /content/HinhAnh --output-dir /content/out --run-id myrun
    python -m client_pipeline prepare --input-dir /content/HinhAnh --output-dir /content/out --run-id myrun
    # Review in the app, pick reject/damage prototypes if needed.
    python -m client_pipeline clean   --output-dir /content/out --run-id myrun --seed --policy
    python -m client_pipeline export  --input-dir /content/HinhAnh --output-dir /content/out --run-id myrun
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from shared.runtime import bootstrap

bootstrap.ensure_repo_root_on_path()


DEFAULT_GDINO_CHECKPOINT = "IDEA-Research/grounding-dino-base"
DEFAULT_OPENCLIP_MODEL = "ViT-H-14"
DEFAULT_OPENCLIP_PRETRAINED = "laion2b_s32b_b79k"
DEFAULT_DINOV2_MODEL = "facebook/dinov2-giant"


def _db_path(args: argparse.Namespace) -> Path:
    raw = str(getattr(args, "db", "") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(args.output_dir).expanduser().resolve() / "pipeline.sqlite3"


def _out_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_dir).expanduser().resolve()


def _run_module(module: str, argv: list[str]) -> int:
    print("\n" + "=" * 88, flush=True)
    print(f"$ python -m {module} {' '.join(argv)}", flush=True)
    print("=" * 88, flush=True)
    mod = importlib.import_module(module)
    return int(mod.main(argv))


def _require_run_id(args: argparse.Namespace) -> str:
    run_id = str(args.run_id or "").strip()
    if not run_id:
        raise SystemExit("--run-id is required for this command.")
    return run_id


def cmd_detect(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = _out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    db = _db_path(args)
    argv = [
        "--input-dir", str(input_dir),
        "--db", str(db),
        "--checkpoint", str(args.gdino_checkpoint),
        "--device", str(args.device),
        "--tiled-threshold", str(args.tiled_threshold),
        "--tile-size", str(args.tile_size),
        "--tile-overlap", str(args.tile_overlap),
        "--tile-batch-size", str(args.tile_batch_size),
        "--nms-iou", str(args.nms_iou),
        "--box-threshold", str(args.box_threshold),
        "--final-max-dets-per-class", str(args.final_max_dets_per_class),
        "--image-workers", str(args.image_workers),
        "--service-workers", str(args.service_workers),
        "--service-queue-size", str(args.service_queue_size),
        "--service-batch-size", str(args.service_batch_size),
        "--store-image-path-mode", str(args.store_image_path_mode),
    ]
    if str(args.service_device_ids or "").strip():
        argv.extend(["--service-device-ids", str(args.service_device_ids)])
    if int(args.limit) > 0:
        argv.extend(["--limit", str(args.limit)])
    if bool(args.recursive):
        argv.append("--recursive")
    if bool(args.save_overlays):
        argv.append("--save-overlays")
    if bool(args.verbose):
        argv.append("--verbose")
    return _run_module("steps.gdino_scan.main", argv)


def cmd_filter(args: argparse.Namespace) -> int:
    run_id = _require_run_id(args)
    input_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = _out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    db = _db_path(args)

    rc = _run_module(
        "steps.openclip_semantic.main",
        [
            "--db", str(db),
            "--image-root", str(input_dir),
            "--source-run-id", str(args.source_run_id),
            "--stage", str(args.stage),
            "--model-name", str(args.openclip_model),
            "--pretrained", str(args.openclip_pretrained),
            "--device", str(args.device),
            "--batch-size", str(args.openclip_batch_size),
            "--num-workers", str(args.openclip_workers),
        ] + (["--limit", str(args.limit)] if int(args.limit) > 0 else []),
    )
    if rc != 0:
        return rc

    return _run_module(
        "steps.step01_semantic.main",
        [
            "--source-db", str(db),
            "--output-db", str(db),
            "--semantic-run-id", str(args.semantic_run_id),
            "--run-id", run_id,
            "--image-root", str(input_dir),
        ] + (["--limit", str(args.limit)] if int(args.limit) > 0 else []),
    )


def cmd_prepare(args: argparse.Namespace) -> int:
    run_id = _require_run_id(args)
    input_dir = Path(args.input_dir).expanduser().resolve()
    db = _db_path(args)
    crop_views = str(args.crop_views)
    view_name = str(args.view_name)
    rc = _run_module(
        "steps.step02_crops.main",
        [
            "--db", str(db),
            "--run-id", run_id,
            "--image-root", str(input_dir),
            "--crop-views", crop_views,
            "--num-workers", str(args.crop_workers),
            "--crop-compress-level", str(args.crop_compress_level),
        ] + (["--limit", str(args.limit)] if int(args.limit) > 0 else []),
    )
    if rc != 0:
        return rc

    embed_argv = [
        "--db", str(db),
        "--run-id", run_id,
        "--model-name", str(args.dinov2_model),
        "--device", str(args.device),
        "--view-name", view_name,
        "--batch-size", str(args.embed_batch_size),
        "--num-workers", str(args.embed_workers),
        "--log-every", str(args.embed_log_every),
    ]
    if bool(args.resume_embed):
        embed_argv.append("--resume")
    else:
        embed_argv.append("--force")
    if int(args.limit) > 0:
        embed_argv.extend(["--limit", str(args.limit)])
    rc = _run_module("steps.step03_embed.main", embed_argv)
    if rc != 0:
        return rc

    if not bool(args.skip_core):
        rc = _run_module(
            "steps.step04_core.main",
            [
                "--db", str(db),
                "--run-id", run_id,
                "--model-name", str(args.dinov2_model),
                "--view-name", view_name,
                "--core-min-size", str(args.core_min_size),
            ],
        )
    return rc


def cmd_clean(args: argparse.Namespace) -> int:
    run_id = _require_run_id(args)
    db = _db_path(args)
    rc = 0
    if bool(args.seed):
        rc = _run_module("tools.relabel_semantic_seed", ["--db", str(db), "--run-id", run_id, "--apply"])
        if rc != 0:
            return rc
    if bool(args.policy):
        rc = _run_module("steps.step06_reliability.main", ["--db", str(db), "--run-id", run_id])
        if rc != 0:
            return rc
        rc = _run_module("steps.step07_decision.main", ["--db", str(db), "--run-id", run_id])
    if not bool(args.seed) and not bool(args.policy):
        print("Nothing to clean. Pass --seed and/or --policy.", flush=True)
    return rc


def cmd_export(args: argparse.Namespace) -> int:
    run_id = _require_run_id(args)
    input_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = _out_dir(args)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve() if str(args.dataset_dir or "").strip() else out_dir / "dataset" / run_id
    db = _db_path(args)
    argv = [
        "--db", str(db),
        "--run-id", run_id,
        "--image-root", str(input_dir),
        "--output-dir", str(dataset_dir),
        "--format", str(args.format),
        "--split", str(args.split),
        "--copy-images",
        "--random-state", str(args.random_state),
    ]
    if bool(args.no_copy_images):
        argv.remove("--copy-images")
    return _run_module("tools.export_dataset", argv)


def cmd_status(args: argparse.Namespace) -> int:
    run_id = _require_run_id(args)
    return _run_module("run_pipeline", ["--db", str(_db_path(args)), "status", "--run-id", run_id])


def cmd_recommended(args: argparse.Namespace) -> int:
    """Run the automatic recommended portion: detect -> filter -> prepare.

    Review/export remain separate so the user can inspect labels before producing
    the final dataset.
    """
    rc = cmd_detect(args)
    if rc != 0:
        return rc
    rc = cmd_filter(args)
    if rc != 0:
        return rc
    return cmd_prepare(args)


def add_common_io(parser: argparse.ArgumentParser, *, input_required: bool = True) -> None:
    parser.add_argument("--input-dir", required=input_required, help="Folder of source images.")
    parser.add_argument("--output-dir", default="/content/out", help="Output root. Default: /content/out")
    parser.add_argument("--db", default="", help="Override SQLite path. Default: <output-dir>/pipeline.sqlite3")
    parser.add_argument("--run-id", default="", help="Client run id, e.g. project_v1.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--limit", type=int, default=0, help="Debug limit. 0 = all.")


def add_detect_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gdino-checkpoint", default=DEFAULT_GDINO_CHECKPOINT)
    parser.add_argument("--tiled-threshold", type=int, default=1024)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=128)
    parser.add_argument("--tile-batch-size", type=int, default=8,
                        help="Tiles/ROIs per GDINO forward. 8 is a safe A100 default; try 12/16 if VRAM is low-utilized.")
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--box-threshold", type=float, default=0.12)
    parser.add_argument("--final-max-dets-per-class", type=int, default=400)
    parser.add_argument("--image-workers", type=int, default=2,
                        help="Images processed concurrently. 2 overlaps CPU/IO with the GPU worker without changing results.")
    parser.add_argument("--service-workers", type=int, default=0,
                        help="GDINO worker processes. 0=auto, usually one per visible GPU.")
    parser.add_argument("--service-queue-size", type=int, default=0)
    parser.add_argument("--service-batch-size", type=int, default=0)
    parser.add_argument("--service-device-ids", default="")
    parser.add_argument("--store-image-path-mode", default="relative", choices=["name", "relative", "absolute"])
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--save-overlays", action="store_true", help="Debug only: save GDINO overlay images. Off by default.")
    parser.add_argument("--verbose", action="store_true")


def add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-run-id", default="latest")
    parser.add_argument("--stage", default="final")
    parser.add_argument("--semantic-run-id", default="latest")
    parser.add_argument("--openclip-model", default=DEFAULT_OPENCLIP_MODEL)
    parser.add_argument("--openclip-pretrained", default=DEFAULT_OPENCLIP_PRETRAINED)
    parser.add_argument("--openclip-batch-size", type=int, default=512)
    parser.add_argument("--openclip-workers", type=int, default=32)


def add_prepare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--crop-views", default="tight")
    parser.add_argument("--view-name", default="tight")
    parser.add_argument("--crop-workers", type=int, default=8)
    parser.add_argument("--crop-compress-level", type=int, default=1)
    parser.add_argument("--dinov2-model", default=DEFAULT_DINOV2_MODEL)
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--embed-workers", type=int, default=8)
    parser.add_argument("--embed-log-every", type=int, default=1024)
    parser.add_argument("--resume-embed", action="store_true", help="Resume a matching embedding run. Default creates a fresh run for this client run.")
    parser.add_argument("--skip-core", action="store_true")
    parser.add_argument("--core-min-size", type=int, default=10)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple client pipeline: images -> cleaned dataset.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("detect", help="Step 1: detect candidate damage boxes with GDINO.")
    add_common_io(p, input_required=True)
    add_detect_args(p)
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("filter", help="Step 2: OpenCLIP semantic filter + initial clean/review tables.")
    add_common_io(p, input_required=True)
    add_filter_args(p)
    p.set_defaults(func=cmd_filter)

    p = sub.add_parser("prepare", help="Step 3: crops + DINOv2 embeddings + core clusters.")
    add_common_io(p, input_required=True)
    add_prepare_args(p)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("clean", help="Optional: apply seed vote and/or policy after prototype/review.")
    add_common_io(p, input_required=False)
    p.add_argument("--seed", action="store_true", help="Apply detector+prototype+core seed vote.")
    p.add_argument("--policy", action="store_true", help="Run reliability scoring + decision policy.")
    p.set_defaults(func=cmd_clean)

    p = sub.add_parser("export", help="Step 5: export cleaned labels to YOLO/COCO dataset.")
    add_common_io(p, input_required=True)
    p.add_argument("--dataset-dir", default="", help="Default: <output-dir>/dataset/<run-id>")
    p.add_argument("--format", default="both", choices=["yolo", "coco", "both"])
    p.add_argument("--split", default="0.8,0.1,0.1", help="train,val,test fractions. Use 1,0,0 for no split.")
    p.add_argument("--random-state", type=int, default=17)
    p.add_argument("--no-copy-images", action="store_true", help="Only write labels/annotations; do not copy images.")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("status", help="Show status for a run.")
    add_common_io(p, input_required=False)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("recommended", help="Run detect -> filter -> prepare. Review/export separately.")
    add_common_io(p, input_required=True)
    add_detect_args(p)
    add_filter_args(p)
    add_prepare_args(p)
    p.set_defaults(func=cmd_recommended)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))