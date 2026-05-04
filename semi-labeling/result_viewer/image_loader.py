from __future__ import annotations

import math
from pathlib import Path
from threading import RLock

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QImage, QPixmap

from models import AssignmentRow


def resolve_image_path(row: AssignmentRow, image_root: Path | None) -> Path:
    candidates: list[Path] = []
    rel_path = str(row.image_rel_path or "").strip()
    stored_path = str(row.image_path or "").strip()
    source_input_dir = Path(str(row.source_input_dir or "")).expanduser()
    if image_root is not None:
        root = image_root.expanduser().resolve()
        candidates.append(root / rel_path)
        if stored_path:
            candidates.append(root / Path(stored_path).name)
    if stored_path:
        stored = Path(stored_path).expanduser()
        candidates.append(stored if stored.is_absolute() else source_input_dir / stored_path)
    if rel_path:
        candidates.append(source_input_dir / rel_path)
    if stored_path:
        candidates.append(source_input_dir / Path(stored_path).name)
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()
    if image_root is not None:
        return (image_root.expanduser().resolve() / rel_path).resolve()
    return (source_input_dir / rel_path).expanduser().resolve()


class ImageLoader:
    def __init__(self) -> None:
        self._image_cache: dict[str, QImage] = {}
        self._crop_cache: dict[tuple[int, int, float, int], QImage] = {}
        self._lock = RLock()

    def clear(self) -> None:
        with self._lock:
            self._image_cache.clear()
            self._crop_cache.clear()

    def crop_pixmap(self, row: AssignmentRow, image_root: Path | None, *, padding_ratio: float, image_size: int) -> QPixmap:
        return QPixmap.fromImage(self.crop_image(row, image_root, padding_ratio=padding_ratio, image_size=image_size))

    def crop_image(self, row: AssignmentRow, image_root: Path | None, *, padding_ratio: float, image_size: int) -> QImage:
        cache_key = (int(row.result_id), int(image_size), round(float(padding_ratio), 4), hash(str(image_root)))
        with self._lock:
            cached = self._crop_cache.get(cache_key)
            if cached is not None:
                return QImage(cached)
        image_path = resolve_image_path(row, image_root)
        image_key = str(image_path)
        with self._lock:
            image = self._image_cache.get(image_key)
        if image is None or image.isNull():
            image = QImage(image_key)
            if image.isNull():
                raise FileNotFoundError(f"Cannot load image: {image_path}")
            with self._lock:
                self._image_cache[image_key] = QImage(image)
        pad_x = max(0.0, float(row.x2) - float(row.x1)) * float(padding_ratio)
        pad_y = max(0.0, float(row.y2) - float(row.y1)) * float(padding_ratio)
        x1 = max(0, int(math.floor(float(row.x1) - pad_x)))
        y1 = max(0, int(math.floor(float(row.y1) - pad_y)))
        x2 = min(image.width(), int(math.ceil(float(row.x2) + pad_x)))
        y2 = min(image.height(), int(math.ceil(float(row.y2) + pad_y)))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid crop for result_id={row.result_id}: {(x1, y1, x2, y2)}")
        crop = image.copy(QRect(x1, y1, x2 - x1, y2 - y1))
        scaled = crop.scaledToWidth(max(80, int(image_size)), Qt.SmoothTransformation)
        with self._lock:
            self._crop_cache[cache_key] = QImage(scaled)
        return scaled
