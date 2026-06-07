from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui


class ImageLoadSignals(QtCore.QObject):
    loaded = QtCore.Signal(object, QtGui.QImage)
    failed = QtCore.Signal(object, str)


class _ImageLoadTask(QtCore.QRunnable):
    def __init__(
        self,
        key: object,
        path: str,
        size: int,
        crop_box: tuple[float, float, float, float] | None = None,
    ) -> None:
        super().__init__()
        self.setAutoDelete(False)
        self.key = key
        self.path = path
        self.size = size
        self.crop_box = crop_box
        self.signals = ImageLoadSignals()

    def run(self) -> None:
        try:
            image = QtGui.QImage(self.path)
            if image.isNull():
                raise ValueError(f"Cannot decode image: {self.path}")
            if self.crop_box is not None:
                x1, y1, x2, y2 = self.crop_box
                left = max(0, min(int(round(x1)), image.width() - 1))
                top = max(0, min(int(round(y1)), image.height() - 1))
                right = max(left + 1, min(int(round(x2)), image.width()))
                bottom = max(top + 1, min(int(round(y2)), image.height()))
                image = image.copy(left, top, right - left, bottom - top)
            if int(self.size or 0) > 0:
                image = image.scaled(
                    self.size,
                    self.size,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
            try:
                self.signals.loaded.emit(self.key, image)
            except RuntimeError:
                # The owning widget/application may have been closed while a
                # background decode was still running.  In that case there is
                # nothing useful left to notify, and printing a traceback makes
                # fast review/close workflows look broken.
                return
        except Exception as exc:
            try:
                self.signals.failed.emit(self.key, str(exc))
            except RuntimeError:
                return


class ImageService(QtCore.QObject):
    imageLoaded = QtCore.Signal(object, QtGui.QPixmap)
    imageFailed = QtCore.Signal(object, str)

    def __init__(
        self,
        max_items: int = 1200,
        max_full_items: int = 8,
        max_thumb_items: int = 900,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._cache: OrderedDict[object, QtGui.QPixmap] = OrderedDict()
        self._loading: set[object] = set()
        self._active_tasks: dict[object, _ImageLoadTask] = {}
        self._max_items = max_items
        self._max_full_items = max_full_items
        self._max_thumb_items = max_thumb_items
        self._pool = QtCore.QThreadPool.globalInstance()

    def cached(self, key: object) -> QtGui.QPixmap | None:
        pixmap = self._cache.get(key)
        if pixmap is not None:
            self._cache.move_to_end(key)
        return pixmap

    def load_image(
        self,
        key: object,
        path: str | Path,
        size: int = 0,
        crop_box: tuple[float, float, float, float] | None = None,
    ) -> None:
        if key in self._cache or key in self._loading:
            return
        raw_path = str(path or "")
        if not raw_path:
            return
        self._loading.add(key)
        task = _ImageLoadTask(key, raw_path, size, crop_box=crop_box)
        task.signals.loaded.connect(self._on_loaded)
        task.signals.failed.connect(self._on_failed)
        self._active_tasks[key] = task
        self._pool.start(task)

    def load_thumbnail(self, key: object, path: str | Path, size: int = 160) -> None:
        self.load_image(key, path, size=size)

    def load_item_thumbnail(
        self,
        key: object,
        path: str | Path,
        box: tuple[float, float, float, float] | None,
        size: int = 92,
    ) -> None:
        self.load_image(key, path, size=size, crop_box=box)

    def _on_loaded(self, key: object, image: QtGui.QImage) -> None:
        self._loading.discard(key)
        self._active_tasks.pop(key, None)
        pixmap = QtGui.QPixmap.fromImage(image)
        self._cache[key] = pixmap
        self._cache.move_to_end(key)
        self._trim_cache()
        self.imageLoaded.emit(key, pixmap)

    def _on_failed(self, key: object, message: str) -> None:
        self._loading.discard(key)
        self._active_tasks.pop(key, None)
        self.imageFailed.emit(key, message)

    def _kind(self, key: object) -> str:
        if isinstance(key, tuple) and key:
            return str(key[0])
        return "other"

    def _trim_cache(self) -> None:
        def trim_kind(kind: str, limit: int) -> None:
            count = sum(1 for key in self._cache if self._kind(key) == kind)
            while count > limit:
                for old_key in list(self._cache.keys()):
                    if self._kind(old_key) == kind:
                        self._cache.pop(old_key, None)
                        count -= 1
                        break
                else:
                    break

        trim_kind("full", self._max_full_items)
        trim_kind("thumb", self._max_thumb_items)
        while len(self._cache) > self._max_items:
            self._cache.popitem(last=False)


def item_image_path(item: Any) -> str:
    crop_path = str(getattr(item, "crop_path", "") or "")
    if crop_path and Path(crop_path).is_file():
        return crop_path
    uri = str(getattr(item, "crop_uri", "") or "")
    if uri.startswith("file:"):
        return QtCore.QUrl(uri).toLocalFile()
    return uri
