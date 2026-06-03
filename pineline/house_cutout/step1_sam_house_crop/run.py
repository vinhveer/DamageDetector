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
    # Console Windows mặc định cp1252 không in được tiếng Việt → ép UTF-8.
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
        default_sam_checkpoint,
        ensure_step1_dirs,
    )
    from pineline.house_cutout.step1_sam_house_crop.prompts import (
        DEFAULT_NEGATIVE_QUERIES,
        DEFAULT_POSITIVE_QUERIES,
        normalize_queries,
    )
    from pineline.house_cutout.step1_sam_house_crop.runner import run_step1

    parser = argparse.ArgumentParser(
        description=(
            "house_cutout step1 — cắt nhà bằng GroundingDINO (house +/ window,door -) "
            "và SAM. Hỗ trợ ảnh .tif lớn."
        ),
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Thư mục ảnh gốc (gồm .tif).")
    parser.add_argument("--db", type=Path, default=STEP1_DB)
    parser.add_argument("--work-dir", type=Path, default=STEP1_WORK_DIR)
    parser.add_argument("--cutouts-dir", type=Path, default=STEP1_CUTOUTS_DIR)
    parser.add_argument("--masks-dir", type=Path, default=STEP1_MASKS_DIR)
    parser.add_argument("--overlay-dir", type=Path, default=STEP1_OVERLAY_DIR)
    parser.add_argument("--summary-csv", type=Path, default=STEP1_SUMMARY_CSV)
    parser.add_argument("--no-overlay", dest="write_overlays", action="store_false")
    parser.set_defaults(write_overlays=True)

    parser.add_argument("--positive", action="append", default=[],
                        help="Query positive (house). Lặp lại được. Mặc định: house/building/...")
    parser.add_argument("--negative", action="append", default=[],
                        help="Query negative (window/door). Lặp lại được. Mặc định: window/door/...")

    parser.add_argument("--checkpoint", type=str, default="",
                        help="GroundingDINO checkpoint. Rỗng = auto-detect.")
    parser.add_argument("--sam-checkpoint", type=Path, default=default_sam_checkpoint())
    parser.add_argument("--sam-model-type", type=str, default="auto")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--work-max-side", type=int, default=4096,
                        help="Downscale working RGB nếu cạnh dài lớn hơn ngưỡng này.")
    parser.add_argument("--box-threshold", type=float, default=0.15)
    parser.add_argument("--text-threshold", type=float, default=0.15)
    parser.add_argument("--max-dets", type=int, default=50)
    parser.add_argument("--tiled-threshold", type=int, default=2048,
                        help="Ảnh working có cạnh > ngưỡng → recursive_detect tìm house.")
    parser.add_argument("--score-floor", type=float, default=0.20,
                        help="Bỏ box (house/negative) dưới ngưỡng điểm này.")
    parser.add_argument("--points-per-box", type=int, default=5)
    parser.add_argument("--pad-px", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="0 = không giới hạn.")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    ensure_step1_dirs()
    positive = normalize_queries(args.positive, DEFAULT_POSITIVE_QUERIES)
    negative = normalize_queries(args.negative, DEFAULT_NEGATIVE_QUERIES)

    res = run_step1(
        input_dir=args.input_dir,
        db_path=args.db,
        work_dir=args.work_dir,
        cutouts_dir=args.cutouts_dir,
        masks_dir=args.masks_dir,
        overlay_dir=args.overlay_dir,
        summary_csv=args.summary_csv,
        write_overlays=args.write_overlays,
        positive_queries=positive,
        negative_queries=negative,
        gdino_checkpoint=args.checkpoint,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        device=args.device,
        work_max_side=args.work_max_side,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        max_dets=args.max_dets,
        tiled_threshold=args.tiled_threshold,
        score_floor=args.score_floor,
        points_per_box=args.points_per_box,
        pad_px=args.pad_px,
        limit=args.limit,
        skip_existing=args.skip_existing,
    )
    print(res, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
