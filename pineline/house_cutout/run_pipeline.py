from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "object_detection").exists() and (parent / "inference_api").exists():
            return parent
    return here.parents[2]


def main(argv: list[str] | None = None) -> int:
    """Chạy step1 (cắt nhà) rồi step2 (detect damage) một lệnh.

    Chỉ phơi ra các tham số dùng chung quan trọng; cần tinh chỉnh sâu thì gọi
    từng step CLI riêng (xem SPEC.md §7).
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    repo_root = _resolve_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from pineline.house_cutout.common.paths import (
        DEFAULT_INPUT_DIR,
        STEP1_CUTOUTS_DIR,
        STEP1_DB,
        STEP1_MASKS_DIR,
        STEP1_OVERLAY_DIR,
        STEP1_SUMMARY_CSV,
        STEP1_WORK_DIR,
        STEP2_DB,
        STEP2_OVERLAY_DIR,
        STEP2_RGB_DIR,
        STEP2_SUMMARY_CSV,
        default_sam_checkpoint,
        ensure_dirs,
    )
    from pineline.house_cutout.step1_sam_house_crop.prompts import (
        DEFAULT_NEGATIVE_QUERIES,
        DEFAULT_POSITIVE_QUERIES,
        normalize_queries,
    )
    from pineline.house_cutout.step1_sam_house_crop.runner import run_step1
    from pineline.house_cutout.step2_gdino_detect.prompts import parse_prompt_groups
    from pineline.house_cutout.step2_gdino_detect.runner import run_step1 as run_step2

    parser = argparse.ArgumentParser(
        description="house_cutout — chạy step1 (cắt nhà SAM+GDINO) → step2 (multi-model detect + segmentation).",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="GroundingDINO checkpoint (dùng cho cả 2 step). Rỗng = auto.")
    parser.add_argument("--detection-models", type=str, default="gdino,yolo,stabledino",
                        help="Step2 detector list: gdino,yolo,stabledino.")
    parser.add_argument("--gdino-tile-batch-size", type=int, default=0,
                        help="Step2 tiles per GDINO forward in recursive mode. 0 = engine/env default.")
    parser.add_argument("--gdino-service-workers", type=int, default=0,
                        help="Step2 DINO worker processes. 0 = auto. For one 8GB GPU, prefer 1.")
    parser.add_argument("--gdino-service-queue-size", type=int, default=0)
    parser.add_argument("--gdino-service-batch-size", type=int, default=0)
    parser.add_argument("--gdino-service-device-ids", type=str, default="")
    parser.add_argument("--yolo-model", type=str, default="")
    parser.add_argument("--stabledino-checkpoint", type=str, default="")
    parser.add_argument("--sam-checkpoint", type=Path, default=default_sam_checkpoint())
    parser.add_argument("--sam-model-type", type=str, default="auto")
    parser.add_argument("--no-segmentation", dest="run_segmentation", action="store_false")
    parser.set_defaults(run_segmentation=True)
    parser.add_argument("--sam-segment-checkpoint", type=str, default="")
    parser.add_argument("--sam-segment-model-type", type=str, default="vit_b")
    parser.add_argument("--unet-model", type=str, default="")
    parser.add_argument("--sam-finetune-checkpoint", type=str, default="")
    parser.add_argument("--sam-finetune-delta-type", type=str, default="lora")
    parser.add_argument("--sam-finetune-delta-checkpoint", type=str, default="")
    parser.add_argument("--sam-finetune-model-type", type=str, default="vit_b")
    parser.add_argument("--positive", action="append", default=[])
    parser.add_argument("--negative", action="append", default=[])
    parser.add_argument("--work-max-side", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-overlay", dest="write_overlays", action="store_false")
    parser.set_defaults(write_overlays=True)
    args = parser.parse_args(argv)

    ensure_dirs()
    positive = normalize_queries(args.positive, DEFAULT_POSITIVE_QUERIES)
    negative = normalize_queries(args.negative, DEFAULT_NEGATIVE_QUERIES)

    print("=== STEP 1: cắt nhà (GDINO + SAM) ===", flush=True)
    res1 = run_step1(
        input_dir=args.input_dir,
        db_path=STEP1_DB,
        work_dir=STEP1_WORK_DIR,
        cutouts_dir=STEP1_CUTOUTS_DIR,
        masks_dir=STEP1_MASKS_DIR,
        overlay_dir=STEP1_OVERLAY_DIR,
        summary_csv=STEP1_SUMMARY_CSV,
        write_overlays=args.write_overlays,
        positive_queries=positive,
        negative_queries=negative,
        gdino_checkpoint=args.checkpoint,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        device=args.device,
        work_max_side=args.work_max_side,
        limit=args.limit,
    )
    print(res1, flush=True)

    if int(res1.get("cutout_count") or 0) <= 0:
        print("Không có cutout nào được tạo ở step1; bỏ qua step2.", flush=True)
        return 0

    print("\n=== STEP 2: detect damage trên cutout ===", flush=True)
    res2 = run_step2(
        input_dir=STEP1_CUTOUTS_DIR,
        db_path=STEP2_DB,
        rgb_dir=STEP2_RGB_DIR,
        overlay_dir=STEP2_OVERLAY_DIR,
        summary_csv=STEP2_SUMMARY_CSV,
        write_overlays=args.write_overlays,
        prompt_groups=parse_prompt_groups([]),
        checkpoint=args.checkpoint,
        detection_models=args.detection_models,
        yolo_model=args.yolo_model or None,
        stabledino_checkpoint=args.stabledino_checkpoint or None,
        device=args.device,
        gdino_tile_batch_size=args.gdino_tile_batch_size,
        gdino_service_workers=args.gdino_service_workers,
        gdino_service_queue_size=args.gdino_service_queue_size,
        gdino_service_batch_size=args.gdino_service_batch_size,
        gdino_service_device_ids=args.gdino_service_device_ids or None,
        run_segmentation=args.run_segmentation,
        sam_segment_checkpoint=args.sam_segment_checkpoint or None,
        sam_segment_model_type=args.sam_segment_model_type,
        unet_model=args.unet_model or None,
        sam_finetune_checkpoint=args.sam_finetune_checkpoint or None,
        sam_finetune_delta_type=args.sam_finetune_delta_type,
        sam_finetune_delta_checkpoint=args.sam_finetune_delta_checkpoint or None,
        sam_finetune_model_type=args.sam_finetune_model_type,
    )
    print(res2, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
