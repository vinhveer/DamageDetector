from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Ensure repo root is importable when running as `python tools/...`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _iter_images(input_dir: Path, *, recursive: bool) -> list[Path]:
    if recursive:
        paths = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
    else:
        paths = [p for p in input_dir.glob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
    return sorted(paths)


def _max_dim_fast(image_path: Path) -> int:
    # Avoid importing cv2 just for dim read.
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
    # BGR for cv2; deterministic by label string.
    h = 2166136261
    for ch in (label or "object").encode("utf-8"):
        h ^= ch
        h *= 16777619
        h &= 0xFFFFFFFF
    r = 40 + (h & 0x7F)
    g = 40 + ((h >> 8) & 0x7F)
    b = 40 + ((h >> 16) & 0x7F)
    return int(b), int(g), int(r)


@dataclass(frozen=True)
class Row:
    image: str
    image_path: str
    label: str
    score: float
    x1: int
    y1: int
    x2: int
    y2: int
    w: int
    h: int
    area_px2: int


def _rows_from_detections(
    *,
    image_path: Path,
    detections: Iterable[dict[str, Any]],
    width: int,
    height: int,
) -> list[Row]:
    rows: list[Row] = []
    for det in detections:
        box = _sanitize_xyxy(det.get("box"), width=width, height=height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        w = int(x2 - x1)
        h = int(y2 - y1)
        rows.append(
            Row(
                image=image_path.name,
                image_path=str(image_path),
                label=str(det.get("label") or ""),
                score=float(det.get("score") or 0.0),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                w=w,
                h=h,
                area_px2=int(w * h),
            )
        )
    return rows


def _write_overlay(
    *,
    image_path: Path,
    overlay_path: Path,
    detections: Iterable[dict[str, Any]],
) -> tuple[int, int, int]:
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    height, width = img.shape[:2]
    overlay = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    drawn = 0
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
        drawn += 1

    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_path), overlay)
    return width, height, drawn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch GroundingDINO detect on a folder; output CSV + overlay images.")
    parser.add_argument("--input-dir", required=True, help="Folder containing images.")
    parser.add_argument("--output-dir", default="results_gdino_folder", help="Where to save CSV + overlays.")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="GroundingDINO checkpoint path or HF model folder/id. If empty, use repo default.",
    )
    parser.add_argument(
        "--queries",
        default="damage, defect, crack, peeling, mold, stain, broken wall, decay",
        help="Comma-separated text queries for GroundingDINO.",
    )
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--max-dets", type=int, default=30)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--recursive-find", action="store_true", help="Also scan subfolders for images.")
    parser.add_argument("--verbose-logs", action="store_true", help="Stream GroundingDINO/tiled logs to stdout.")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N images (0 = all).")
    parser.add_argument(
        "--tiled-threshold",
        type=int,
        default=512,
        help="If max(image_width,height) > this, use recursive/tiled DINO. Default: 512.",
    )
    parser.add_argument(
        "--tile-scales",
        default="small,medium,large",
        help="In tiled mode, which scales to run: comma-separated from {small,medium,large}. Default: small,medium,large.",
    )
    parser.add_argument("--recursive-max-depth", type=int, default=3)
    parser.add_argument("--recursive-min-box-px", type=int, default=48)
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    out_dir = Path(args.output_dir).expanduser().resolve()
    overlays_dir = out_dir / "overlays"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)

    from dino.client import get_dino_service
    from dino.engine import default_gdino_checkpoint

    service = get_dino_service()
    try:
        images = _iter_images(input_dir, recursive=bool(args.recursive_find))
        if int(args.limit or 0) > 0:
            images = images[: int(args.limit)]
        if not images:
            print(f"No images found in {input_dir}", flush=True)
            return 2

        checkpoint = str(args.checkpoint or "").strip() or str(default_gdino_checkpoint() or "").strip()
        if not checkpoint:
            raise RuntimeError("No GroundingDINO checkpoint available. Pass --checkpoint or download the default model.")

        params = {
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

        all_rows: list[Row] = []
        for idx, img_path in enumerate(images, start=1):
            max_dim = _max_dim_fast(img_path)
            use_tiled = int(max_dim) > int(args.tiled_threshold)
            print(f"[{idx}/{len(images)}] start {img_path.name} mode={'tiled' if use_tiled else 'single'}", flush=True)
            log_fn = (lambda s: print(s, flush=True)) if bool(args.verbose_logs) else None
            if use_tiled:
                result = service.call(
                    "recursive_detect",
                    {
                        "image_path": str(img_path),
                        "params": params,
                        "target_labels": params["text_queries"],
                        "max_depth": int(args.recursive_max_depth),
                        "min_box_px": int(args.recursive_min_box_px),
                    },
                    log_fn=log_fn,
                )
            else:
                result = service.call("predict", {"image_path": str(img_path), "params": params}, log_fn=log_fn)

            # Use the filtered detections (after containment filter + NMS + max_dets).
            # `display_detections` can be extremely large in tiled mode.
            detections = list(result.get("detections") or [])

            # Image dims for CSV sanitization:
            from PIL import Image

            with Image.open(img_path) as pil_img:
                w, h = pil_img.size

            rows = _rows_from_detections(image_path=img_path, detections=detections, width=int(w), height=int(h))
            all_rows.extend(rows)

            overlay_path = overlays_dir / f"{img_path.stem}_gdino.png"
            _write_overlay(image_path=img_path, overlay_path=overlay_path, detections=detections)

            print(f"[{idx}/{len(images)}] done {img_path.name}: dets={len(rows)} overlay={overlay_path.name}", flush=True)

        csv_path = out_dir / "detections.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "image",
                    "image_path",
                    "label",
                    "score",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "w",
                    "h",
                    "area_px2",
                ]
            )
            for row in all_rows:
                writer.writerow(
                    [
                        row.image,
                        row.image_path,
                        row.label,
                        f"{row.score:.6f}",
                        row.x1,
                        row.y1,
                        row.x2,
                        row.y2,
                        row.w,
                        row.h,
                        row.area_px2,
                    ]
                )

        print(f"Saved CSV: {csv_path}", flush=True)
        print(f"Saved overlays: {overlays_dir}", flush=True)
        return 0
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
