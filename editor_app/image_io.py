from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QImageReader

QImageReader.setAllocationLimit(0)


class ImageIoError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedMask:
    mask: QImage


def load_image(path: str | Path) -> QImage:
    image = QImage(str(path))
    if image.isNull():
        raise ImageIoError(f"Không mở được ảnh: {path}")
    if image.format() != QImage.Format_ARGB32_Premultiplied:
        image = image.convertToFormat(QImage.Format_ARGB32_Premultiplied)
    return image


def _threshold_to_binary_grayscale(image: QImage) -> QImage:
    gray = image.convertToFormat(QImage.Format_Grayscale8)
    width, height = gray.width(), gray.height()
    bytes_per_line = gray.bytesPerLine()
    buffer = gray.bits()
    array = np.frombuffer(buffer, dtype=np.uint8).reshape((height, bytes_per_line))
    array[:, :width] = np.where(array[:, :width] > 0, 255, 0).astype(np.uint8)
    return gray


def load_mask(path: str | Path, expected_size: tuple[int, int]) -> LoadedMask:
    mask = QImage(str(path))
    if mask.isNull():
        raise ImageIoError(f"Không mở được mask: {path}")

    target_w, target_h = expected_size
    if (mask.width(), mask.height()) != (target_w, target_h):
        mask = mask.scaled(target_w, target_h, Qt.IgnoreAspectRatio, Qt.FastTransformation)
    return LoadedMask(mask=_threshold_to_binary_grayscale(mask))


def new_blank_mask(size: tuple[int, int]) -> LoadedMask:
    width, height = size
    mask = QImage(width, height, QImage.Format_Grayscale8)
    mask.fill(0)
    return LoadedMask(mask=mask)


def save_mask_png_01_indexed(path: str | Path, mask: QImage) -> None:
    if mask.isNull():
        raise ImageIoError("Mask đang rỗng")
    if mask.format() != QImage.Format_Grayscale8:
        raise ImageIoError("Mask phải là QImage.Format_Grayscale8")

    width, height = mask.width(), mask.height()
    out = QImage(width, height, QImage.Format_Indexed8)
    out.setColorTable([QColor(0, 0, 0).rgba(), QColor(255, 255, 255).rgba()])
    out.fill(0)

    src_buf = mask.constBits()
    src_bpl = mask.bytesPerLine()
    dst_buf = out.bits()
    dst_bpl = out.bytesPerLine()

    for y in range(height):
        src_row = src_buf[y * src_bpl : y * src_bpl + width]
        dst_row = dst_buf[y * dst_bpl : y * dst_bpl + width]
        for x in range(width):
            dst_row[x] = 1 if src_row[x] else 0

    if not out.save(str(path)):
        raise ImageIoError(f"Lưu mask thất bại: {path}")


def save_mask_png_0255(path: str | Path, mask: QImage) -> None:
    if mask.isNull():
        raise ImageIoError("Mask đang rỗng")
    if mask.format() != QImage.Format_Grayscale8:
        raise ImageIoError("Mask phải là QImage.Format_Grayscale8")
    if not mask.save(str(path)):
        raise ImageIoError(f"Lưu mask thất bại: {path}")
