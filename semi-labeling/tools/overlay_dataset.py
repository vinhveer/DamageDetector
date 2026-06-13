#!/usr/bin/env python3
"""Draw cleaned_labels boxes onto their source images for visual QA.

Reads cleaned_labels (result_id, image_rel_path, final_label, x1..y2) for a run
and writes one overlay PNG per image with per-class coloured boxes + labels.

Usage:
    python -m tools.overlay_dataset --db DB --run-id myrun \
        --image-root /path/HinhAnh --output-dir /path/overlay
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from shared.runtime import bootstrap

bootstrap.ensure_on_path()

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from shared.db.schema import connect_output  # noqa: E402

LABEL_COLORS = {
    "crack": (74, 144, 217),    # blue
    "mold": (39, 174, 96),      # green
    "spall": (243, 156, 18),    # orange
    "reject": (231, 76, 60),    # red
}
DEFAULT_COLOR = (127, 140, 141)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw cleaned_labels boxes onto source images.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Debug: only first N images. 0 = all.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_root = Path(args.image_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = connect_output(args.db)
    try:
        rows = conn.execute(
            """SELECT image_rel_path, final_label, x1, y1, x2, y2
               FROM cleaned_labels WHERE run_id=? AND x1 IS NOT NULL
               ORDER BY image_rel_path""",
            (str(args.run_id),),
        ).fetchall()
    finally:
        conn.close()

    by_image: dict[str, list] = defaultdict(list)
    for row in rows:
        by_image[str(row["image_rel_path"])].append(row)

    try:
        font = ImageFont.load_default(16)
    except Exception:
        font = ImageFont.load_default()

    images = sorted(by_image)
    if int(args.limit) > 0:
        images = images[: int(args.limit)]

    written = 0
    missing = 0
    total_boxes = 0
    for rel in images:
        src = image_root / rel
        if not src.is_file():
            missing += 1
            continue
        try:
            img = Image.open(src).convert("RGB")
        except Exception:
            missing += 1
            continue
        draw = ImageDraw.Draw(img)
        for row in by_image[rel]:
            label = str(row["final_label"] or "")
            color = LABEL_COLORS.get(label, DEFAULT_COLOR)
            x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
            draw.rectangle([x1, y1, x2, y2], outline=color, width=int(args.line_width))
            tx, ty = x1, max(0.0, y1 - 18)
            try:
                tb = draw.textbbox((tx, ty), label, font=font)
                draw.rectangle(tb, fill=color)
            except Exception:
                pass
            draw.text((tx, ty), label, fill=(255, 255, 255), font=font)
            total_boxes += 1
        # Mirror the source path so collisions across subfolders are avoided.
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            dst = dst.with_suffix(".png")
        img.save(dst)
        written += 1

    print(f"overlay_written={written}  missing_source={missing}  boxes_drawn={total_boxes}")
    print(f"output_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
