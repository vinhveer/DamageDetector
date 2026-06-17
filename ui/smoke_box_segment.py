from __future__ import annotations

import argparse
from pathlib import Path

from ui.services.box_segment_pipeline import MODEL_ROOT, run_box_segment_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test UI box-scoped segmentation.")
    parser.add_argument("--image", default="/Users/nguyenquangvinh/Downloads/wall-crack.jpg")
    parser.add_argument("--output-dir", default=str(MODEL_ROOT / "ui_smoke_box_segmentation"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--detector", choices=["yolo", "manual"], default="yolo")
    parser.add_argument("--yolo-conf", type=float, default=0.10)
    parser.add_argument("--segmenters", default="sam_lora_hq_coarse_refine,unet")
    parser.add_argument("--max-boxes", type=int, default=3)
    args = parser.parse_args()

    segmenters = [part.strip() for part in str(args.segmenters).split(",") if part.strip()]
    summary = run_box_segment_smoke(
        Path(args.image),
        Path(args.output_dir),
        device=args.device,
        detector=args.detector,
        yolo_conf=float(args.yolo_conf),
        segmenters=segmenters,
        max_boxes=int(args.max_boxes),
        log_fn=print,
    )
    summary_path = Path(str(summary["output_dir"])) / "summary.json"
    print(f"summary={summary_path}")
    print(f"boxes={len(summary.get('boxes') or [])}")
    failed = [name for name, payload in (summary.get("segmenters") or {}).items() if isinstance(payload, dict) and payload.get("error")]
    if failed:
        print(f"failed={failed}")
        return 1
    for name, payload in (summary.get("segmenters") or {}).items():
        stats = payload.get("stats") or []
        print(f"{name}: masks={len(stats)}")
        for item in stats:
            print(
                f"  area_px={item['area_px']} outside_px={item['outside_px']} "
                f"outside_ratio={item['outside_ratio']:.4f} full_image_ratio={item['full_image_ratio']:.4f} "
                f"box_fill_ratio={item['box_fill_ratio']:.4f} path={item['mask_path']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
