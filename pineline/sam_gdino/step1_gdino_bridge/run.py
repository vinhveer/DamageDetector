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
    repo_root = _resolve_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from pineline.sam_gdino.common.paths import (
        STEP1_DB, STEP1_OVERLAY_DIR, STEP1_SUMMARY_CSV, ensure_dirs,
    )
    from pineline.sam_gdino.step1_gdino_bridge.runner import run_step1

    parser = argparse.ArgumentParser(
        description="Step 1 — GroundingDINO bridge detect with dynamic threshold.",
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=STEP1_DB)
    parser.add_argument("--overlay-dir", type=Path, default=STEP1_OVERLAY_DIR,
                        help="Where to write per-image overlay PNGs (kept images only).")
    parser.add_argument("--summary-csv", type=Path, default=STEP1_SUMMARY_CSV,
                        help="Where to write a flat CSV of all kept detections.")
    parser.add_argument("--no-overlay", dest="write_overlays", action="store_false")
    parser.set_defaults(write_overlays=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--box-threshold", type=float, default=0.10,
                        help="Raw GDINO threshold (low to gather candidates).")
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--score-floor", type=float, default=0.20,
                        help="Drop image if best box score < floor.")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Keep top-K boxes per image after sorting by score.")
    parser.add_argument("--max-dets", type=int, default=25)
    parser.add_argument("--tiled-threshold", type=int, default=1600)
    parser.add_argument("--recursive-find", action="store_true", default=True)
    parser.add_argument("--no-recursive-find", dest="recursive_find", action="store_false")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    ensure_dirs()
    res = run_step1(
        input_dir=args.input_dir,
        db_path=args.db,
        overlay_dir=args.overlay_dir,
        summary_csv=args.summary_csv,
        write_overlays=args.write_overlays,
        checkpoint=args.checkpoint,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        score_floor=args.score_floor,
        top_k=args.top_k,
        max_dets=args.max_dets,
        tiled_threshold=args.tiled_threshold,
        recursive_find=args.recursive_find,
        limit=args.limit,
    )
    print(res, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
