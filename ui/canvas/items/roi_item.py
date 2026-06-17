from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class RoiRectItem(QtWidgets.QGraphicsObject):
    """Selectable, movable ROI rectangle that emits positionChanged when dragged."""

    # Emitted after drag: (roi_index, pos_before, pos_after) in scene coords
    positionChanged = QtCore.Signal(int, QtCore.QPointF, QtCore.QPointF)

    def __init__(self, rect: QtCore.QRectF, roi_index: int) -> None:
        super().__init__()
        self.roi_index = int(roi_index)
        self._rect = QtCore.QRectF(rect)
        self._drag_start_pos: QtCore.QPointF | None = None
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setZValue(2)
        self._selected = False

    def rect(self) -> QtCore.QRectF:
        return QtCore.QRectF(self._rect)

    def setRect(self, rect: QtCore.QRectF) -> None:
        self.prepareGeometryChange()
        self._rect = QtCore.QRectF(rect)
        self.update()

    def boundingRect(self) -> QtCore.QRectF:
        return self._rect.adjusted(-2, -2, 2, 2)

    def shape(self) -> QtGui.QPainterPath:
        path = QtGui.QPainterPath()
        path.addRect(self._rect)
        return path

    def scene_rect(self) -> QtCore.QRectF:
        """Convenience: rect in scene coordinates."""
        return self.mapRectToScene(self._rect)

    def _color(self) -> QtGui.QColor:
        return QtGui.QColor(255, 120, 80) if self._selected else QtGui.QColor(255, 198, 41)

    def paint(self, painter: QtGui.QPainter, option: QtWidgets.QStyleOptionGraphicsItem, widget=None) -> None:  # noqa: ANN001
        color = self._color()
        pen = QtGui.QPen(color, 3 if self._selected else 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(color.red(), color.green(), color.blue(), 45 if self._selected else 30)))
        painter.drawRect(self._rect)

        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        font = painter.font()
        font.setBold(True)
        font.setPointSizeF(max(10.0, self._rect.height() * 0.08))
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtGui.QColor(20, 20, 20)))
        tag = f"#{self.roi_index}"
        pos = self._rect.topLeft() + QtCore.QPointF(6.0, max(16.0, font.pointSizeF() + 4.0))
        label_rect = QtCore.QRectF(self._rect.left(), self._rect.top(), 14.0 + 12.0 * len(tag), font.pointSizeF() + 10.0)
        painter.fillRect(label_rect, QtGui.QColor(255, 198, 41, 200))
        painter.drawText(pos, tag)
        painter.restore()

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_start_pos = self.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        super().mouseReleaseEvent(event)
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._drag_start_pos is not None:
            start = self._drag_start_pos
            end = self.pos()
            self._drag_start_pos = None
            if (end - start).manhattanLength() > 1.0:
                self.positionChanged.emit(self.roi_index, start, end)

    def itemChange(self, change: QtWidgets.QGraphicsItem.GraphicsItemChange, value):  # noqa: ANN001
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._selected = bool(value)
            self.update()
        return super().itemChange(change, value)
