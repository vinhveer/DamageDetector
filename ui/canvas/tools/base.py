from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class Tool(QtCore.QObject):
    """Base class for canvas tools.

    A tool owns event handling on the canvas. Only one tool is active at a time.
    Returning True from a handler marks the event as consumed.
    """

    name: str = ""
    label: str = ""
    shortcut: str = ""
    cursor: QtCore.Qt.CursorShape = QtCore.Qt.CursorShape.ArrowCursor

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._canvas: QtWidgets.QGraphicsView | None = None

    def activate(self, canvas: QtWidgets.QGraphicsView) -> None:
        self._canvas = canvas
        canvas.setCursor(self.cursor)

    def deactivate(self) -> None:
        if self._canvas is not None:
            self._canvas.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        self._canvas = None

    def options_widget(self) -> QtWidgets.QWidget | None:
        return None

    def mouse_press(self, event: QtGui.QMouseEvent, scene_pos: QtCore.QPointF) -> bool:
        return False

    def mouse_move(self, event: QtGui.QMouseEvent, scene_pos: QtCore.QPointF) -> bool:
        return False

    def mouse_release(self, event: QtGui.QMouseEvent, scene_pos: QtCore.QPointF) -> bool:
        return False

    def key_press(self, event: QtGui.QKeyEvent) -> bool:
        return False
