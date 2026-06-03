from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui


class ImageLoadSignals(QtCore.QObject):
    loaded = QtCore.Signal(object, QtGui.QImage)
    failed = QtCore.Signal(object, str)


class _ImageLoadTask(QtCore.QRunnable):
    def __init__(self, key: object, path: str, size: int) -> None:
        super().__init__()
        self.key = key
        self.path = path
        self.size = size
        self.signals = ImageLoadSignals()

    def run(self) -> None:
        try:
            image = QtGui.QImage(self.path)
            if image.isNull():
                raise ValueError(f"Cannot decode image: {self.path}")
            image = image.scaled(
                self.size,
                self.size,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self.signals.loaded.emit(self.key, image)
        except Exception as exc:
            self.signals.failed.emit(self.key, str(exc))


class ImageService(QtCore.QObject):
    imageLoaded = QtCore.Signal(object, QtGui.QPixmap)
    imageFailed = QtCore.Signal(object, str)

    def __init__(self, max_items: int = 1200, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._cache: OrderedDict[object, QtGui.QPixmap] = OrderedDict()
        self._loading: set[object] = set()
        self._max_items = max_items
        self._pool = QtCore.QThreadPool.globalInstance()

    def cached(self, key: object) -> QtGui.QPixmap | None:
        pixmap = self._cache.get(key)
        if pixmap is not None:
            self._cache.move_to_end(key)
        return pixmap

    def load_thumbnail(self, key: object, path: str | Path, size: int = 160) -> None:
        if key in self._cache or key in self._loading:
            return
        raw_path = str(path or "")
        if not raw_path:
            return
        self._loading.add(key)
        task = _ImageLoadTask(key, raw_path, size)
        task.signals.loaded.connect(self._on_loaded)
        task.signals.failed.connect(self._on_failed)
        self._pool.start(task)

    def _on_loaded(self, key: object, image: QtGui.QImage) -> None:
        self._loading.discard(key)
        pixmap = QtGui.QPixmap.fromImage(image)
        self._cache[key] = pixmap
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_items:
            self._cache.popitem(last=False)
        self.imageLoaded.emit(key, pixmap)

    def _on_failed(self, key: object, message: str) -> None:
        self._loading.discard(key)
        self.imageFailed.emit(key, message)


def item_image_path(item: Any) -> str:
    crop_path = str(getattr(item, "crop_path", "") or "")
    if crop_path and Path(crop_path).is_file():
        return crop_path
    uri = str(getattr(item, "crop_uri", "") or "")
    if uri.startswith("file:"):
        return QtCore.QUrl(uri).toLocalFile()
    return uri
