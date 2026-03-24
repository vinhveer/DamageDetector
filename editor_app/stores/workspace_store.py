from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui


class WorkspaceStore(QtCore.QObject):
    workspaceChanged = QtCore.Signal()
    imageChanged = QtCore.Signal()
    maskChanged = QtCore.Signal()
    detectionsChanged = QtCore.Signal()
    highlightChanged = QtCore.Signal()
    roiChanged = QtCore.Signal()

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.workspace_root: Path | None = None
        self.results_root: Path | None = None
        self.images: list[str] = []
        self.current_index: int = -1
        self.current_image_path: str | None = None
        self.current_mask_path: str | None = None
        self.current_image = QtGui.QImage()
        self.current_mask = QtGui.QImage()
        self.current_detections: list[dict] = []
        self.highlight_detections: list[dict] = []
        self._rois_by_image: dict[str, list[tuple[int, int, int, int]]] = {}
        self.current_rois: list[tuple[int, int, int, int]] = []
        self.current_roi_index: int = -1

    def set_workspace(self, workspace_root: Path | None, results_root: Path | None, images: list[str]) -> None:
        self.workspace_root = workspace_root
        self.results_root = results_root
        self.images = list(images)
        self.current_index = -1 if not self.current_image_path else self.images.index(self.current_image_path) if self.current_image_path in self.images else -1
        self.workspaceChanged.emit()

    def set_current_image(self, path: str, image: QtGui.QImage) -> None:
        self.current_image_path = str(path)
        self.current_image = QtGui.QImage(image)
        self.current_index = self.images.index(self.current_image_path) if self.current_image_path in self.images else -1
        self.current_rois = list(self._rois_by_image.get(self.current_image_path, []))
        self.current_roi_index = 0 if self.current_rois else -1
        self.imageChanged.emit()
        self.roiChanged.emit()

    def set_current_mask(self, path: str | None, mask: QtGui.QImage) -> None:
        self.current_mask_path = str(path) if path else None
        self.current_mask = QtGui.QImage(mask)
        self.maskChanged.emit()

    def clear_mask(self, mask: QtGui.QImage) -> None:
        self.current_mask_path = None
        self.current_mask = QtGui.QImage(mask)
        self.maskChanged.emit()

    def set_detections(self, detections: list[dict]) -> None:
        self.current_detections = [dict(det) for det in detections]
        self.detectionsChanged.emit()

    def set_highlight_detections(self, detections: list[dict]) -> None:
        self.highlight_detections = [dict(det) for det in detections]
        self.highlightChanged.emit()

    def current_roi_box(self) -> tuple[int, int, int, int] | None:
        if self.current_roi_index < 0 or self.current_roi_index >= len(self.current_rois):
            return None
        return tuple(int(x) for x in self.current_rois[self.current_roi_index])

    def select_roi_index(self, index: int) -> None:
        if index < 0 or index >= len(self.current_rois):
            index = -1
        if index == self.current_roi_index:
            return
        self.current_roi_index = index
        self.roiChanged.emit()

    def add_roi(self, roi_box: tuple[int, int, int, int]) -> None:
        if not self.current_image_path:
            return
        roi = tuple(int(x) for x in roi_box)
        rois = list(self._rois_by_image.get(self.current_image_path, []))
        if roi in rois:
            self.current_rois = list(rois)
            self.current_roi_index = rois.index(roi)
            self.roiChanged.emit()
            return
        rois.append(roi)
        self._rois_by_image[self.current_image_path] = rois
        self.current_rois = list(rois)
        self.current_roi_index = len(rois) - 1
        self.roiChanged.emit()

    def update_roi(self, index: int, roi_box: tuple[int, int, int, int]) -> None:
        if not self.current_image_path:
            return
        rois = list(self._rois_by_image.get(self.current_image_path, []))
        if index < 0 or index >= len(rois):
            return
        rois[index] = tuple(int(x) for x in roi_box)
        self._rois_by_image[self.current_image_path] = rois
        self.current_rois = list(rois)
        self.current_roi_index = index
        self.roiChanged.emit()

    def remove_roi(self, index: int) -> None:
        if not self.current_image_path:
            return
        rois = list(self._rois_by_image.get(self.current_image_path, []))
        if index < 0 or index >= len(rois):
            return
        rois.pop(index)
        self._rois_by_image[self.current_image_path] = rois
        self.current_rois = list(rois)
        if not rois:
            self.current_roi_index = -1
        else:
            self.current_roi_index = min(index, len(rois) - 1)
        self.roiChanged.emit()

    def clear_rois(self) -> None:
        if not self.current_image_path:
            return
        self._rois_by_image[self.current_image_path] = []
        self.current_rois = []
        self.current_roi_index = -1
        self.roiChanged.emit()
