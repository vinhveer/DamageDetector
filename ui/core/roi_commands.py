from __future__ import annotations

from PySide6 import QtCore

from ui.core.commands import Command
from ui.widgets.canvas import ImageCanvas


class AddRoiCommand(Command):
    label = "Add ROI"

    def __init__(self, canvas: ImageCanvas, rect: QtCore.QRectF) -> None:
        self._canvas = canvas
        self._rect = QtCore.QRectF(rect)
        self._roi_index: int | None = None

    def redo(self) -> None:
        item = self._canvas.add_roi_rect(self._rect)
        self._roi_index = int(item.roi_index)

    def undo(self) -> None:
        if self._roi_index is not None:
            self._canvas.delete_roi(self._roi_index)


class DeleteRoiCommand(Command):
    label = "Delete ROI"

    def __init__(self, canvas: ImageCanvas, roi_index: int, rect: QtCore.QRectF) -> None:
        self._canvas = canvas
        self._roi_index = int(roi_index)
        self._rect = QtCore.QRectF(rect)

    def redo(self) -> None:
        self._canvas.delete_roi(self._roi_index)

    def undo(self) -> None:
        item = self._canvas.add_roi_rect(self._rect)
        self._roi_index = int(item.roi_index)


class MoveRoiCommand(Command):
    """Undo/redo for dragging a ROI rectangle to a new position."""

    label = "Move ROI"

    def __init__(self, canvas: ImageCanvas, roi_index: int, before: QtCore.QPointF, after: QtCore.QPointF) -> None:
        self._canvas = canvas
        self._roi_index = int(roi_index)
        self._before = QtCore.QPointF(before)
        self._after = QtCore.QPointF(after)

    def _find_item(self):  # noqa: ANN201
        for item in self._canvas.roi_items():
            if item.roi_index == self._roi_index:
                return item
        return None

    def redo(self) -> None:
        item = self._find_item()
        if item is not None:
            item.setPos(self._after)

    def undo(self) -> None:
        item = self._find_item()
        if item is not None:
            item.setPos(self._before)
