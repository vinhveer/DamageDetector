from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ui.canvas.tools.base import Tool


class PanTool(Tool):
    name = "pan"
    label = "Pan"
    shortcut = "H"
    cursor = QtCore.Qt.CursorShape.OpenHandCursor

    def activate(self, canvas: QtWidgets.QGraphicsView) -> None:
        super().activate(canvas)
        canvas.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)

    def deactivate(self) -> None:
        if self._canvas is not None:
            self._canvas.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        super().deactivate()


class SelectTool(Tool):
    name = "select"
    label = "Select"
    shortcut = "V"
    cursor = QtCore.Qt.CursorShape.ArrowCursor

    def activate(self, canvas: QtWidgets.QGraphicsView) -> None:
        super().activate(canvas)
        canvas.setDragMode(QtWidgets.QGraphicsView.DragMode.RubberBandDrag)

    def deactivate(self) -> None:
        if self._canvas is not None:
            self._canvas.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        super().deactivate()


class RectRoiTool(Tool):
    name = "rect_roi"
    label = "Rect ROI"
    shortcut = "R"
    cursor = QtCore.Qt.CursorShape.CrossCursor

    roiCommitted = QtCore.Signal(QtCore.QRectF)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._drag_start: QtCore.QPointF | None = None
        self._rubber: QtWidgets.QGraphicsRectItem | None = None

    def activate(self, canvas: QtWidgets.QGraphicsView) -> None:
        super().activate(canvas)
        canvas.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)

    def _scene(self) -> QtWidgets.QGraphicsScene | None:
        return self._canvas.scene() if self._canvas is not None else None

    def _bounds(self) -> QtCore.QRectF:
        scene = self._scene()
        if scene is None:
            return QtCore.QRectF()
        return scene.sceneRect()

    def mouse_press(self, event: QtGui.QMouseEvent, scene_pos: QtCore.QPointF) -> bool:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return False
        scene = self._scene()
        if scene is None or self._bounds().isEmpty():
            return False
        self._drag_start = scene_pos
        pen = QtGui.QPen(QtGui.QColor(255, 198, 41), 2, QtCore.Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self._rubber = scene.addRect(QtCore.QRectF(scene_pos, scene_pos), pen)
        self._rubber.setZValue(50)
        return True

    def mouse_move(self, event: QtGui.QMouseEvent, scene_pos: QtCore.QPointF) -> bool:
        if self._drag_start is None or self._rubber is None:
            return False
        rect = QtCore.QRectF(self._drag_start, scene_pos).normalized()
        rect = rect.intersected(self._bounds())
        self._rubber.setRect(rect)
        return True

    def mouse_release(self, event: QtGui.QMouseEvent, scene_pos: QtCore.QPointF) -> bool:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return False
        if self._rubber is None or self._scene() is None:
            return False
        rect = self._rubber.rect().normalized().intersected(self._bounds())
        self._scene().removeItem(self._rubber)
        self._rubber = None
        self._drag_start = None
        if rect.width() >= 8 and rect.height() >= 8:
            self.roiCommitted.emit(rect)
        return True

    def key_press(self, event: QtGui.QKeyEvent) -> bool:
        if event.key() == QtCore.Qt.Key.Key_Escape and self._rubber is not None:
            scene = self._scene()
            if scene is not None:
                scene.removeItem(self._rubber)
            self._rubber = None
            self._drag_start = None
            return True
        return False
