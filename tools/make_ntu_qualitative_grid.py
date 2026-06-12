"""Build 4x4 qualitative montages (YOLO left / StableDINO right) for NTU images.

Two figure families:
  * Chapter III: crack-only overlays already rendered under
    crack_object_detection/selected_hinhanh_conf020_set2 (copied as-is).
  * Chapter IV: multi-class overlays re-drawn from boxes JSON under
    semi_labeling_training/selected_hinhanh_conf020_semilabeling with the
    chapter-IV palette (crack=blue, mold=green, spall=orange) plus a legend.

Each montage packs 8 ROIs of one cluster into a 4x4 grid: the left two columns
are YOLO, the right two columns are StableDINO, four rows of ROIs. Every cell is
labelled with its ROI id and each half carries a column title.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


GROUP_1 = "cum_1_it_ngoai_canh"
GROUP_2 = "cum_2_co_ngoai_canh"

# Chapter IV class palette (RGB).
CLASS_COLORS = {
    "crack": (33, 150, 243),
    "mold": (76, 175, 80),
    "spall": (255, 152, 0),
}
FALLBACK_COLOR = (255, 87, 34)

CELL = 520            # rendered size of each overlay cell (square canvas)
PAD = 16              # padding around cells
LABEL_H = 40          # ROI label strip height
TITLE_H = 60          # column title strip height
LEGEND_H = 60         # legend strip height (chapter IV only)
GUTTER = 28           # gap between YOLO half and StableDINO half
BG = (255, 255, 255)
INK = (20, 20, 20)


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _load_manifest(model_root: Path) -> list[dict[str, Any]]:
    path = model_root / "inputs" / "selected_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _fit(image: Image.Image, size: int) -> Image.Image:
    canvas = Image.new("RGB", (size, size), BG)
    work = image.convert("RGB")
    work.thumbnail((size, size), Image.LANCZOS)
    off = ((size - work.width) // 2, (size - work.height) // 2)
    canvas.paste(work, off)
    return canvas


def _draw_boxes(src_path: Path, boxes: list[dict[str, Any]]) -> Image.Image:
    image = Image.open(src_path).convert("RGB")
    width, _ = image.size
    draw = ImageDraw.Draw(image)
    line_w = max(3, width // 260)
    font = _font(max(18, width // 38))
    for box in sorted(boxes, key=lambda b: float(b.get("score", 0.0))):
        label = str(box.get("label", "crack"))
        score = float(box.get("score", 0.0))
        x1, y1, x2, y2 = (float(v) for v in box["xyxy"])
        color = CLASS_COLORS.get(label, FALLBACK_COLOR)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
        text = f"{label} {score:.2f}"
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0] + 8, tb[3] - tb[1] + 6
        yt = max(0, y1 - th)
        draw.rectangle([x1, yt, x1 + tw, yt + th], fill=color)
        draw.text((x1 + 4, yt + 2), text, fill=(255, 255, 255), font=font)
    return image


def _labelled_cell(cell_img: Image.Image, roi: int) -> Image.Image:
    canvas = Image.new("RGB", (CELL, CELL + LABEL_H), BG)
    canvas.paste(cell_img, (0, LABEL_H))
    draw = ImageDraw.Draw(canvas)
    font = _font(26)
    text = f"roi{roi}"
    tb = draw.textbbox((0, 0), text, font=font)
    draw.text(((CELL - (tb[2] - tb[0])) // 2, (LABEL_H - (tb[3] - tb[1])) // 2 - tb[1]),
              text, fill=INK, font=font)
    return canvas


def _cell_yolo_stable(roi_item: dict[str, Any], yolo_img: Image.Image,
                      stable_img: Image.Image) -> tuple[Image.Image, Image.Image]:
    left = _labelled_cell(_fit(yolo_img, CELL), int(roi_item["roi"]))
    right = _labelled_cell(_fit(stable_img, CELL), int(roi_item["roi"]))
    return left, right


def _build_grid(cells: list[tuple[Image.Image, Image.Image]], legend: bool) -> Image.Image:
    cell_h = CELL + LABEL_H
    half_w = 2 * CELL + PAD
    grid_w = PAD + half_w + GUTTER + half_w + PAD
    grid_h = TITLE_H + PAD + 4 * (cell_h + PAD) + (LEGEND_H if legend else 0)
    canvas = Image.new("RGB", (grid_w, grid_h), BG)
    draw = ImageDraw.Draw(canvas)

    title_font = _font(34)
    left_x0 = PAD
    right_x0 = PAD + half_w + GUTTER
    for title, x0 in (("YOLO26x", left_x0), ("StableDINO", right_x0)):
        tb = draw.textbbox((0, 0), title, font=title_font)
        draw.text((x0 + (half_w - (tb[2] - tb[0])) // 2,
                   (TITLE_H - (tb[3] - tb[1])) // 2 - tb[1]),
                  title, fill=INK, font=title_font)

    for idx, (left, right) in enumerate(cells):
        row = idx // 2
        col = idx % 2
        y = TITLE_H + PAD + row * (cell_h + PAD)
        lx = left_x0 + col * CELL
        rx = right_x0 + col * CELL
        canvas.paste(left, (lx, y))
        canvas.paste(right, (rx, y))

    if legend:
        ly = grid_h - LEGEND_H + 14
        lx = PAD
        sw = 30
        legend_font = _font(26)
        for label, color in CLASS_COLORS.items():
            draw.rectangle([lx, ly, lx + sw, ly + sw], fill=color, outline=INK)
            draw.text((lx + sw + 8, ly + 2), label, fill=INK, font=legend_font)
            tb = draw.textbbox((0, 0), label, font=legend_font)
            lx += sw + 16 + (tb[2] - tb[0]) + 40
    return canvas


def _order(manifest: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    return [item for item in manifest if item["group"] == group]


def build_chapter3(out_dir: Path) -> list[Path]:
    root = workspace_root() / "model_with_inference" / "crack_object_detection" / "selected_hinhanh_conf020_set2"
    manifest = _load_manifest(root)
    yolo_dir = root / "yolo_26x_img768" / "overlays"
    stable_dir = root / "stable_dino_r50_img768" / "overlays"
    written: list[Path] = []
    for group, suffix in ((GROUP_1, "it_ngoai_canh"), (GROUP_2, "co_ngoai_canh")):
        cells = []
        for item in _order(manifest, group):
            name = item["image"]
            yolo_img = Image.open(yolo_dir / group / name)
            stable_img = Image.open(stable_dir / group / name)
            cells.append(_cell_yolo_stable(item, yolo_img, stable_img))
        grid = _build_grid(cells, legend=False)
        out = out_dir / f"ch3_ntu_crack_qualitative_{suffix}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        grid.save(out)
        written.append(out)
        print(f"wrote {out}")
    return written


def _boxes_index(model_root: Path, model_name: str) -> dict[str, list[dict[str, Any]]]:
    path = model_root / model_name / "boxes" / "predictions_conf020.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["image"]: row.get("boxes", []) for row in rows}


def build_chapter4(out_dir: Path) -> list[Path]:
    root = workspace_root() / "model_with_inference" / "semi_labeling_training" / "selected_hinhanh_conf020_semilabeling"
    manifest = _load_manifest(root)
    yolo_boxes = _boxes_index(root, "yolo_26x_img768")
    stable_boxes = _boxes_index(root, "stable_dino_r50_img768")
    written: list[Path] = []
    for group, suffix in ((GROUP_1, "it_ngoai_canh"), (GROUP_2, "co_ngoai_canh")):
        cells = []
        for item in _order(manifest, group):
            name = item["image"]
            src = Path(item["input_path"])
            yolo_img = _draw_boxes(src, yolo_boxes.get(name, []))
            stable_img = _draw_boxes(src, stable_boxes.get(name, []))
            cells.append(_cell_yolo_stable(item, yolo_img, stable_img))
        grid = _build_grid(cells, legend=True)
        out = out_dir / f"ch4_ntu_multiclass_qualitative_{suffix}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        grid.save(out)
        written.append(out)
        print(f"wrote {out}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build NTU qualitative montages")
    parser.add_argument("--chapter", choices=["ch3", "ch4", "all"], default="all")
    args = parser.parse_args(argv)
    thesis = workspace_root() / "DoAnTotNghiep_NguyenQuangVinh_64132989" / "material" / "figures"
    if args.chapter in ("ch3", "all"):
        build_chapter3(thesis / "ch3")
    if args.chapter in ("ch4", "all"):
        build_chapter4(thesis / "ch4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
