from __future__ import annotations

from PySide6 import QtCore

from image_io import load_image, load_mask


class ImageIoWorker(QtCore.QObject):
    failed = QtCore.Signal(str)
    finished = QtCore.Signal(object)

    def __init__(self, kind: str, path: str, expected_size: tuple[int, int] | None = None) -> None:
        super().__init__()
        self._kind = str(kind)
        self._path = str(path)
        self._expected_size = expected_size

    @QtCore.Slot()
    def run(self) -> None:
        try:
            if self._kind == "image":
                self.finished.emit({"kind": "image", "path": self._path, "image": load_image(self._path)})
                return
            if self._kind == "mask":
                if self._expected_size is None:
                    raise RuntimeError("Missing expected mask size.")
                loaded = load_mask(self._path, self._expected_size)
                self.finished.emit({"kind": "mask", "path": self._path, "mask": loaded.mask})
                return
            raise RuntimeError(f"Unsupported IO worker kind: {self._kind}")
        except Exception as exc:
            self.failed.emit(str(exc))
