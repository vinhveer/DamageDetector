from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from ui.create_data_tools.cropper_app.domain import Roi


@dataclass(frozen=True)
class ExportResult:
    exported: int
    skipped: int


def export_rois(*, base_dir: Path, out_dir: Path, items: list[tuple[Path, list[Roi]]]) -> ExportResult:
    base_dir = Path(base_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped = 0
    for image_path, rois in items:
        image_path = Path(image_path)
        if not image_path.is_file():
            skipped += len(rois)
            continue

        try:
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)
        except Exception:
            skipped += len(rois)
            continue

        stem = image_path.stem
        for roi in rois:
            x0 = max(0, int(roi.x))
            y0 = max(0, int(roi.y))
            x1 = max(x0 + 1, int(roi.x + roi.size))
            y1 = max(y0 + 1, int(roi.y + roi.size))
            x1 = min(x1, img.width)
            y1 = min(y1, img.height)
            if x1 <= x0 or y1 <= y0:
                skipped += 1
                continue

            crop = img.crop((x0, y0, x1, y1))
            out_name = f"{stem}__roi{roi.id}.png"
            out_path = out_dir / out_name
            try:
                crop.save(out_path)
                exported += 1
            except Exception:
                skipped += 1

    return ExportResult(exported=exported, skipped=skipped)
