from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtCore, QtGui, QtWidgets


@dataclass(frozen=True)
class RoiGeometry:
    x: int
    y: int
    size: int


class _RoiHandleItem(QtWidgets.QGraphicsRectItem):
    def __init__(self, name: str, parent: "SquareRoiItem") -> None:
        super().__init__(-6.0, -6.0, 12.0, 12.0, parent)
        self._name = str(name)
        self._parent = parent
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setAcceptHoverEvents(True)
        self.setZValue(10.0)

        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 1)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QtGui.QBrush(QtGui.QColor(255, 80, 80, 240)))

        self._dragging = False

    def hoverMoveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:
        del event
        self.setCursor(QtCore.Qt.CursorShape.SizeFDiagCursor)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._dragging = True
            self._parent.setSelected(True)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if self._dragging:
            self._parent._handle_drag(self._name, event.scenePos())  # noqa: SLF001
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        was_dragging = self._dragging
        self._dragging = False
        super().mouseReleaseEvent(event)
        if was_dragging:
            self._parent.geometryCommitted.emit(self._parent.roi_id, self._parent.geometry())


class SquareRoiItem(QtWidgets.QGraphicsObject):
    geometryCommitted = QtCore.Signal(int, object)  # roi_id, RoiGeometry

    def __init__(self, roi_id: int, *, x: int, y: int, size: int, image_bounds: QtCore.QRectF) -> None:
        super().__init__()
        self.roi_id = int(roi_id)
        self.setAcceptHoverEvents(True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        self._image_bounds = QtCore.QRectF(image_bounds)
        self._min_size = 12
        self._rect = QtCore.QRectF(0.0, 0.0, float(size), float(size))
        self._handles = {
            "tl": _RoiHandleItem("tl", self),
            "tr": _RoiHandleItem("tr", self),
            "bl": _RoiHandleItem("bl", self),
            "br": _RoiHandleItem("br", self),
        }
        self.setPos(float(x), float(y))
        self._clamp_into_bounds()
        self._sync_handles()
        self._set_handles_visible(False)

    def set_image_bounds(self, bounds: QtCore.QRectF) -> None:
        self._image_bounds = QtCore.QRectF(bounds)
        self._clamp_into_bounds()
        self._sync_handles()
        self.update()

    def geometry(self) -> RoiGeometry:
        return RoiGeometry(
            x=int(round(self.pos().x())),
            y=int(round(self.pos().y())),
            size=int(round(self._rect.width())),
        )

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(self._rect)

    def paint(self, painter: QtGui.QPainter, option: QtWidgets.QStyleOptionGraphicsItem, widget=None) -> None:  # noqa: ANN001
        del option, widget
        painter.save()
        pen_color = QtGui.QColor(50, 220, 120, 230)
        fill_color = QtGui.QColor(50, 220, 120, 35)
        if self.isSelected():
            pen_color = QtGui.QColor(255, 80, 80, 240)
            fill_color = QtGui.QColor(255, 80, 80, 35)
        pen = QtGui.QPen(pen_color, 4)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(QtGui.QBrush(fill_color))
        painter.drawRect(self._rect)
        painter.restore()

    def itemChange(self, change: QtWidgets.QGraphicsItem.GraphicsItemChange, value):  # noqa: ANN001
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            if hasattr(self, "_handles"):
                self._set_handles_visible(bool(value))
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            new_pos = QtCore.QPointF(value)
            return self._clamp_pos(new_pos)
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if hasattr(self, "_handles"):
                self._sync_handles()
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        was_moving = event.button() == QtCore.Qt.MouseButton.LeftButton
        super().mouseReleaseEvent(event)
        if was_moving:
            self.geometryCommitted.emit(self.roi_id, self.geometry())

    def hoverMoveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:
        del event
        self.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)

    def _set_handles_visible(self, visible: bool) -> None:
        for handle in self._handles.values():
            handle.setVisible(bool(visible))

    def _sync_handles(self) -> None:
        r = self._rect
        self._handles["tl"].setPos(r.left(), r.top())
        self._handles["tr"].setPos(r.right(), r.top())
        self._handles["bl"].setPos(r.left(), r.bottom())
        self._handles["br"].setPos(r.right(), r.bottom())

    def _clamp_pos(self, pos: QtCore.QPointF) -> QtCore.QPointF:
        bounds = QtCore.QRectF(self._image_bounds)
        if bounds.isNull() or bounds.width() <= 0 or bounds.height() <= 0:
            return pos
        size = float(self._rect.width())
        x = max(bounds.left(), min(float(pos.x()), bounds.right() - size))
        y = max(bounds.top(), min(float(pos.y()), bounds.bottom() - size))
        return QtCore.QPointF(x, y)

    def _clamp_into_bounds(self) -> None:
        bounds = QtCore.QRectF(self._image_bounds)
        if bounds.isNull() or bounds.width() <= 0 or bounds.height() <= 0:
            return
        size = float(self._rect.width())
        size = max(float(self._min_size), min(size, float(bounds.width()), float(bounds.height())))
        self.prepareGeometryChange()
        self._rect = QtCore.QRectF(0.0, 0.0, size, size)
        self.setPos(self._clamp_pos(self.pos()))

    def _handle_drag(self, name: str, scene_pos: QtCore.QPointF) -> None:
        name = str(name)
        start_tl = QtCore.QPointF(self.pos())
        size = float(self._rect.width())
        start_br = start_tl + QtCore.QPointF(size, size)
        start_tr = start_tl + QtCore.QPointF(size, 0.0)
        start_bl = start_tl + QtCore.QPointF(0.0, size)

        if name == "tl":
            anchor = start_br
            new_size = max(anchor.x() - scene_pos.x(), anchor.y() - scene_pos.y())
            new_tl = QtCore.QPointF(anchor.x() - new_size, anchor.y() - new_size)
        elif name == "tr":
            anchor = start_bl
            new_size = max(scene_pos.x() - anchor.x(), anchor.y() - scene_pos.y())
            new_tl = QtCore.QPointF(anchor.x(), anchor.y() - new_size)
        elif name == "bl":
            anchor = start_tr
            new_size = max(anchor.x() - scene_pos.x(), scene_pos.y() - anchor.y())
            new_tl = QtCore.QPointF(anchor.x() - new_size, anchor.y())
        else:  # "br"
            anchor = start_tl
            new_size = max(scene_pos.x() - anchor.x(), scene_pos.y() - anchor.y())
            new_tl = QtCore.QPointF(anchor.x(), anchor.y())

        bounds = QtCore.QRectF(self._image_bounds)
        if bounds.isNull() or bounds.width() <= 0 or bounds.height() <= 0:
            new_size = max(float(self._min_size), float(new_size))
        else:
            new_size = max(float(self._min_size), min(float(new_size), float(bounds.width()), float(bounds.height())))

            if name == "tl":
                new_tl = QtCore.QPointF(max(bounds.left(), new_tl.x()), max(bounds.top(), new_tl.y()))
                max_size = min(anchor.x() - new_tl.x(), anchor.y() - new_tl.y())
                new_size = min(new_size, max_size)
                new_tl = QtCore.QPointF(anchor.x() - new_size, anchor.y() - new_size)
            elif name == "tr":
                new_tl = QtCore.QPointF(min(new_tl.x(), bounds.right() - new_size), max(bounds.top(), new_tl.y()))
                new_tl = QtCore.QPointF(max(bounds.left(), new_tl.x()), new_tl.y())
                max_size = min(bounds.right() - new_tl.x(), anchor.y() - new_tl.y())
                new_size = min(new_size, max_size)
                new_tl = QtCore.QPointF(anchor.x(), anchor.y() - new_size)
            elif name == "bl":
                new_tl = QtCore.QPointF(max(bounds.left(), new_tl.x()), min(new_tl.y(), bounds.bottom() - new_size))
                new_tl = QtCore.QPointF(new_tl.x(), max(bounds.top(), new_tl.y()))
                max_size = min(anchor.x() - new_tl.x(), bounds.bottom() - new_tl.y())
                new_size = min(new_size, max_size)
                new_tl = QtCore.QPointF(anchor.x() - new_size, anchor.y())
            else:  # "br"
                new_tl = QtCore.QPointF(max(bounds.left(), new_tl.x()), max(bounds.top(), new_tl.y()))
                max_size = min(bounds.right() - new_tl.x(), bounds.bottom() - new_tl.y())
                new_size = min(new_size, max_size)

        self.prepareGeometryChange()
        self._rect = QtCore.QRectF(0.0, 0.0, float(new_size), float(new_size))
        self.setPos(self._clamp_pos(new_tl))
        self._sync_handles()


class ImageRoiView(QtWidgets.QGraphicsView):
    roiSelectionChanged = QtCore.Signal(object)  # roi_id | None
    addRoiRequested = QtCore.Signal()
    deleteRequested = QtCore.Signal(int)  # roi_id

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item: QtWidgets.QGraphicsPixmapItem | None = None
        self._rois: dict[int, SquareRoiItem] = {}
        self._image_bounds = QtCore.QRectF()

        self._panning = False
        self._space_panning = False
        self._pan_start = QtCore.QPoint()

        self._scene.selectionChanged.connect(self._emit_selection)

    def clear_image(self) -> None:
        self._scene.clear()
        self._pixmap_item = None
        self._rois.clear()
        self._image_bounds = QtCore.QRectF()

    def set_image_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        self.clear_image()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setPos(0.0, 0.0)
        self._image_bounds = QtCore.QRectF(0.0, 0.0, float(pixmap.width()), float(pixmap.height()))
        self._scene.setSceneRect(self._image_bounds)
        self.fit_to_image()
        self.setFocus()

    def fit_to_image(self) -> None:
        if self._scene.sceneRect().isNull():
            return
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def set_rois(self, items: list[SquareRoiItem]) -> None:
        for roi in self._rois.values():
            self._scene.removeItem(roi)
        self._rois.clear()
        for item in items:
            item.set_image_bounds(self._image_bounds)
            self._rois[item.roi_id] = item
            self._scene.addItem(item)

    def roi_item(self, roi_id: int) -> SquareRoiItem | None:
        return self._rois.get(int(roi_id))

    def selected_roi_id(self) -> int | None:
        selected = self._scene.selectedItems()
        for item in selected:
            if isinstance(item, SquareRoiItem):
                return int(item.roi_id)
        return None

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if event.angleDelta().y() == 0:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            roi_id = self.selected_roi_id()
            if roi_id is not None:
                self.deleteRequested.emit(int(roi_id))
                event.accept()
                return
        if event.key() == QtCore.Qt.Key.Key_Alt and not event.isAutoRepeat():
            self.addRoiRequested.emit()
            event.accept()
            return
        if event.key() == QtCore.Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_panning = True
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        if event.key() == QtCore.Qt.Key.Key_F:
            self.fit_to_image()
            event.accept()
            return
        if event.matches(QtGui.QKeySequence.StandardKey.ZoomIn):
            self.scale(1.25, 1.25)
            event.accept()
            return
        if event.matches(QtGui.QKeySequence.StandardKey.ZoomOut):
            self.scale(0.8, 0.8)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Space and self._space_panning and not event.isAutoRepeat():
            self._space_panning = False
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
        if event.button() in (QtCore.Qt.MouseButton.MiddleButton, QtCore.Qt.MouseButton.RightButton):
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() in (QtCore.Qt.MouseButton.MiddleButton, QtCore.Qt.MouseButton.RightButton) and self._panning:
            self._panning = False
            if self._space_panning:
                self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _emit_selection(self) -> None:
        self.roiSelectionChanged.emit(self.selected_roi_id())
