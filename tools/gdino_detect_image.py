from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is importable when running as `python tools/...`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _max_dim_fast(image_path: Path) -> int:
    from PIL import Image

    with Image.open(image_path) as img:
        w, h = img.size
    return int(max(w, h))


def _sanitize_xyxy(box: Any, *, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    except Exception:
        return None
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _stable_color(label: str) -> tuple[int, int, int]:
    h = 2166136261
    for ch in (label or "object").encode("utf-8"):
        h ^= ch
        h *= 16777619
        h &= 0xFFFFFFFF
    r = 40 + (h & 0x7F)
    g = 40 + ((h >> 8) & 0x7F)
    b = 40 + ((h >> 16) & 0x7F)
    return int(b), int(g), int(r)  # BGR


def _write_overlay(*, image_path: Path, overlay_path: Path, detections: list[dict[str, Any]]) -> None:
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    height, width = img.shape[:2]
    overlay = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        box = _sanitize_xyxy(det.get("box"), width=width, height=height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        label = str(det.get("label") or "object")
        score = float(det.get("score") or 0.0)
        color = _stable_color(label)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}".strip()
        if text:
            cv2.putText(overlay, text, (x1, max(14, y1 - 6)), font, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(overlay, text, (x1, max(14, y1 - 6)), font, 0.55, color, 1, cv2.LINE_AA)

    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_path), overlay)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GroundingDINO on a single image; save CSV row dump + overlay.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="results_gdino_single")
    parser.add_argument("--checkpoint", default="", help="GroundingDINO checkpoint path/folder/id. If empty, use repo default.")
    parser.add_argument(
        "--queries",
        default="damage, defect, crack, peeling, mold, stain, broken wall, decay, spalling",
        help="Comma-separated text queries for GroundingDINO.",
    )
    parser.add_argument("--box-threshold", type=float, default=0.19)
    parser.add_argument("--text-threshold", type=float, default=0.19)
    parser.add_argument("--max-dets", type=int, default=80)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--tiled-threshold", type=int, default=512, help="If max dim > this, use tiled recursive detect.")
    parser.add_argument("--tile-scales", default="small,medium,large", help="In tiled mode, scales: small,medium,large.")
    parser.add_argument("--recursive-max-depth", type=int, default=3)
    parser.add_argument("--recursive-min-box-px", type=int, default=48)
    args = parser.parse_args(argv)

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    out_dir = Path(args.output_dir).expanduser().resolve()
    os.makedirs(out_dir, exist_ok=True)

    from dino.client import get_dino_service
    from dino.engine import default_gdino_checkpoint

    checkpoint = str(args.checkpoint or "").strip() or str(default_gdino_checkpoint() or "").strip()
    if not checkpoint:
        raise RuntimeError("No GroundingDINO checkpoint available. Pass --checkpoint or download the default model.")

    params: dict[str, Any] = {
        "gdino_checkpoint": checkpoint,
        "gdino_config_id": "auto",
        "text_queries": [q.strip() for q in str(args.queries).split(",") if q.strip()],
        "box_threshold": float(args.box_threshold),
        "text_threshold": float(args.text_threshold),
        "max_dets": int(args.max_dets),
        "device": args.device,
        "output_dir": str(out_dir),
        "recursive_tile_scales": [s.strip() for s in str(args.tile_scales).split(",") if s.strip()],
    }

    use_tiled = _max_dim_fast(image_path) > int(args.tiled_threshold)
    service = get_dino_service()
    try:
        if use_tiled:
            result = service.call(
                "recursive_detect",
                {
                    "image_path": str(image_path),
                    "params": params,
                    "target_labels": params["text_queries"],
                    "max_depth": int(args.recursive_max_depth),
                    "min_box_px": int(args.recursive_min_box_px),
                },
            )
        else:
            result = service.call("predict", {"image_path": str(image_path), "params": params})
    finally:
        service.close()

    detections = list((result or {}).get("detections") or [])
    overlay_path = out_dir / f"{image_path.stem}_gdino_overlay.png"
    _write_overlay(image_path=image_path, overlay_path=overlay_path, detections=detections)

    print(f"Saved overlay: {overlay_path}")
    print(f"dets: {len(detections)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

