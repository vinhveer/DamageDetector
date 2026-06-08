from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "object_detection").exists() and (parent / "inference_api").exists():
            return parent
    return here.parents[3]


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    repo_root = _resolve_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from pineline.house_cutout.common.paths import (
        STEP1_CUTOUTS_DIR,
        STEP2_DB,
        STEP2_OVERLAY_DIR,
        STEP2_RGB_DIR,
        STEP2_SUMMARY_CSV,
        ensure_step2_dirs,
    )
    from pineline.house_cutout.step2_gdino_detect.prompts import parse_prompt_groups
    from pineline.house_cutout.step2_gdino_detect.runner import run_step1 as run_step2

    parser = argparse.ArgumentParser(
        description=(
            "house_cutout step2 — multi-model detect damage + routed segmentation "
            "trên cutout nhà từ step1."
        ),
    )
    parser.add_argument("--input-dir", type=Path, default=STEP1_CUTOUTS_DIR,
                        help="Thư mục cutout RGBA (mặc định: cutouts của step1).")
    parser.add_argument("--db", type=Path, default=STEP2_DB)
    parser.add_argument("--rgb-dir", type=Path, default=STEP2_RGB_DIR)
    parser.add_argument("--overlay-dir", type=Path, default=STEP2_OVERLAY_DIR)
    parser.add_argument("--summary-csv", type=Path, default=STEP2_SUMMARY_CSV)
    parser.add_argument("--no-overlay", dest="write_overlays", action="store_false")
    parser.set_defaults(write_overlays=True)
    parser.add_argument("--prompt-group", action="append", default=[],
                        help="'name=q1,q2,...'. Lặp lại được. Mặc định crack/mold/stain.")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--detection-models", type=str, default="gdino,yolo,stabledino",
                        help="Comma list: gdino,yolo,stabledino.")
    parser.add_argument("--yolo-model", type=str, default="")
    parser.add_argument("--stabledino-checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--box-threshold", type=float, default=0.10)
    parser.add_argument("--text-threshold", type=float, default=0.10)
    parser.add_argument("--yolo-conf", type=float, default=0.05)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--stabledino-conf", type=float, default=0.05)
    parser.add_argument("--max-dets", type=int, default=150)
    parser.add_argument("--tiled-threshold", type=int, default=400)
    parser.add_argument("--tile-scales", type=str, default="small,medium")
    parser.add_argument("--recursive-max-depth", type=int, default=2)
    parser.add_argument("--min-box-px", type=int, default=12)
    parser.add_argument("--gdino-tile-batch-size", type=int, default=0,
                        help="Tiles per GDINO forward in recursive mode. 0 = engine/env default.")
    parser.add_argument("--gdino-service-workers", type=int, default=0,
                        help="DINO worker processes. 0 = auto. For one 8GB GPU, prefer 1.")
    parser.add_argument("--gdino-service-queue-size", type=int, default=0,
                        help="DINO service queue size. 0 = auto.")
    parser.add_argument("--gdino-service-batch-size", type=int, default=0,
                        help="Image chunk size for DINO predict_batch. 0 = auto.")
    parser.add_argument("--gdino-service-device-ids", type=str, default="",
                        help="Optional comma-separated CUDA ids for DINO workers, e.g. 0,1.")
    parser.add_argument("--max-box-area-ratio", type=float, default=0.50)
    parser.add_argument("--no-box-quality-filter", dest="box_quality_filter", action="store_false",
                        help="Disable semi-labeling-style geometry cleanup for nested/broad/composite boxes.")
    parser.set_defaults(box_quality_filter=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--source-run-id", type=str, default="")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-segmentation", dest="run_segmentation", action="store_false")
    parser.set_defaults(run_segmentation=True)
    parser.add_argument("--segmentation-output-dir", type=Path, default=None)
    parser.add_argument("--sam-segment-checkpoint", type=str, default="",
                        help="SAM checkpoint for non-crack damage. Default targets SAM ViT-B.")
    parser.add_argument("--sam-segment-model-type", type=str, default="vit_b")
    parser.add_argument("--unet-model", type=str, default="")
    parser.add_argument("--sam-finetune-checkpoint", type=str, default="")
    parser.add_argument("--sam-finetune-delta-type", type=str, default="lora")
    parser.add_argument("--sam-finetune-delta-checkpoint", type=str, default="")
    parser.add_argument("--sam-finetune-model-type", type=str, default="vit_b")
    parser.add_argument("--segmentation-threshold", type=float, default=0.5)
    parser.add_argument("--segmentation-min-score", type=float, default=0.2,
                        help="Only detections with score >= this are sent to segmentation; all detections are still stored.")
    args = parser.parse_args(argv)

    ensure_step2_dirs()
    groups = parse_prompt_groups(args.prompt_group)
    tile_scales = [s.strip() for s in args.tile_scales.split(",") if s.strip()]

    res = run_step2(
        input_dir=args.input_dir,
        db_path=args.db,
        rgb_dir=args.rgb_dir,
        overlay_dir=args.overlay_dir,
        summary_csv=args.summary_csv,
        write_overlays=args.write_overlays,
        prompt_groups=groups,
        checkpoint=args.checkpoint,
        detection_models=args.detection_models,
        yolo_model=args.yolo_model or None,
        stabledino_checkpoint=args.stabledino_checkpoint or None,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        stabledino_conf=args.stabledino_conf,
        max_dets=args.max_dets,
        tiled_threshold=args.tiled_threshold,
        tile_scales=tile_scales,
        recursive_max_depth=args.recursive_max_depth,
        min_box_px=args.min_box_px,
        gdino_tile_batch_size=args.gdino_tile_batch_size,
        gdino_service_workers=args.gdino_service_workers,
        gdino_service_queue_size=args.gdino_service_queue_size,
        gdino_service_batch_size=args.gdino_service_batch_size,
        gdino_service_device_ids=args.gdino_service_device_ids or None,
        max_box_area_ratio=args.max_box_area_ratio,
        box_quality_filter=args.box_quality_filter,
        limit=args.limit,
        source_run_id=(args.source_run_id or None),
        skip_existing=args.skip_existing,
        run_segmentation=args.run_segmentation,
        segmentation_output_dir=args.segmentation_output_dir,
        sam_segment_checkpoint=args.sam_segment_checkpoint or None,
        sam_segment_model_type=args.sam_segment_model_type,
        unet_model=args.unet_model or None,
        sam_finetune_checkpoint=args.sam_finetune_checkpoint or None,
        sam_finetune_delta_type=args.sam_finetune_delta_type,
        sam_finetune_delta_checkpoint=args.sam_finetune_delta_checkpoint or None,
        sam_finetune_model_type=args.sam_finetune_model_type,
        segmentation_threshold=args.segmentation_threshold,
        segmentation_min_score=args.segmentation_min_score,
    )
    print(res, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
