from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets


from ui.canvas.items.roi_item import RoiRectItem
from ui.canvas.items.mask_item import MaskPixmapItem


class ImageCanvas(QtWidgets.QGraphicsView):
    """QGraphicsView that delegates input events to the active Tool."""

    roisChanged = QtCore.Signal()
    cursorMoved = QtCore.Signal(float, float)   # image coords (x, y)
    zoomChanged = QtCore.Signal(float)           # zoom factor 0..N

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        self.setMinimumSize(720, 420)
        self.setMouseTracking(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._scene = QtWidgets.QGraphicsScene(self)
        self._scene.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setScene(self._scene)
        self.setBackgroundRole(QtGui.QPalette.ColorRole.Shadow)
        self._pixmap_item: QtWidgets.QGraphicsPixmapItem | None = None
        self._roi_counter = 0
        self._detections: list[QtWidgets.QGraphicsRectItem] = []
        self._masks: list[MaskPixmapItem] = []
        # Per detection-group canvas items, keyed by group/layer id.
        self._group_boxes: dict[str, list[QtWidgets.QGraphicsRectItem]] = {}
        self._group_masks: dict[str, list[MaskPixmapItem]] = {}
        self._active_tool = None
        self._space_panning = False
        self._prev_tool = None

    def current_zoom(self) -> float:
        return float(self.transform().m11())

    def set_active_tool(self, tool) -> None:  # noqa: ANN001
        if self._active_tool is tool:
            return
        if self._active_tool is not None:
            self._active_tool.deactivate()
        self._active_tool = tool
        if tool is not None:
            tool.activate(self)

    def active_tool(self):  # noqa: ANN201
        return self._active_tool

    # Emitted after drag: (roi_index, pos_before, pos_after)
    roiMoved = QtCore.Signal(int, QtCore.QPointF, QtCore.QPointF)

    def add_roi_rect(self, rect: QtCore.QRectF) -> RoiRectItem:
        bounds = self.image_rect()
        if bounds.isEmpty():
            raise RuntimeError("Cannot add ROI without image")
        rect = rect.normalized().intersected(bounds)
        if rect.width() < 4 or rect.height() < 4:
            raise ValueError("ROI too small")
        self._roi_counter += 1
        item = RoiRectItem(rect, self._roi_counter)
        item.positionChanged.connect(self.roiMoved)
        self._scene.addItem(item)
        self.roisChanged.emit()
        return item

    def set_image(self, path: Path) -> None:
        pixmap = QtGui.QPixmap(str(path))
        if pixmap.isNull():
            raise RuntimeError(f"Cannot load image: {path}")
        self._scene.clear()
        self._detections.clear()
        self._masks.clear()
        self._group_boxes.clear()
        self._group_masks.clear()
        self._roi_counter = 0
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(0)
        self._scene.setSceneRect(QtCore.QRectF(pixmap.rect()))
        self.fit_image()
        self.roisChanged.emit()

    def image_rect(self) -> QtCore.QRectF:
        return QtCore.QRectF() if self._pixmap_item is None else self._pixmap_item.boundingRect()

    def roi_items(self) -> list[RoiRectItem]:
        return [item for item in self._scene.items() if isinstance(item, RoiRectItem)]

    def roi_rects(self) -> list[QtCore.QRectF]:
        rects = []
        bounds = self.image_rect()
        for item in sorted(self.roi_items(), key=lambda r: r.roi_index):
            rect = item.mapRectToScene(item.rect()).normalized().intersected(bounds)
            if rect.width() >= 4 and rect.height() >= 4:
                rects.append(rect)
        return rects

    def clear_rois(self) -> None:
        for item in self.roi_items():
            self._scene.removeItem(item)
        self.clear_detections()
        self.roisChanged.emit()

    def delete_roi(self, roi_index: int) -> None:
        for item in self.roi_items():
            if item.roi_index == int(roi_index):
                self._scene.removeItem(item)
        self.roisChanged.emit()

    def delete_selected_rois(self) -> int:
        removed = 0
        for item in self.roi_items():
            if item.isSelected():
                self._scene.removeItem(item)
                removed += 1
        if removed:
            self.roisChanged.emit()
        return removed

    def select_roi(self, roi_index: int) -> None:
        for item in self.roi_items():
            selected = item.roi_index == int(roi_index)
            item.setSelected(selected)
            if selected:
                self.centerOn(item)

    def fit_image(self) -> None:
        if self._pixmap_item is not None:
            self.fitInView(self._scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.zoomChanged.emit(self.current_zoom())

    _GROUP_COLORS = {
        "crack": QtGui.QColor(55, 150, 255),
        "mold": QtGui.QColor(52, 211, 153),
        "stain": QtGui.QColor(245, 158, 11),
        "spall": QtGui.QColor(248, 113, 113),
    }

    def clear_detections(self) -> None:
        for item in self._detections:
            self._scene.removeItem(item)
        self._detections.clear()

    def clear_masks(self) -> None:
        for item in self._masks:
            self._scene.removeItem(item)
        self._masks.clear()
        for items in self._group_masks.values():
            for item in items:
                self._scene.removeItem(item)
        self._group_masks.clear()

    def add_mask_item(self, mask_item: MaskPixmapItem) -> None:
        self._scene.addItem(mask_item)
        self._masks.append(mask_item)

    def mask_items(self) -> list[MaskPixmapItem]:
        items = list(self._masks)
        for group_items in self._group_masks.values():
            items.extend(group_items)
        return items

    def set_masks_visible(self, visible: bool) -> None:
        for item in self.mask_items():
            item.setVisible(visible)

    def set_masks_opacity(self, opacity: float) -> None:
        for item in self.mask_items():
            item.setOpacity(float(opacity))

    # ------------------------------------------------------------------ groups

    def _clear_group_boxes(self, group_id: str) -> None:
        for item in self._group_boxes.pop(group_id, []):
            self._scene.removeItem(item)

    def render_detection_groups(self, groups: list[Any], active_group_id: str | None) -> None:
        """Redraw boxes for all detection groups.

        Each group's visibility is read from its `visible` attribute; the active
        group is emphasised with a thicker, opaque pen.
        """
        for group_id in list(self._group_boxes.keys()):
            self._clear_group_boxes(group_id)
        for group in groups:
            group_id = getattr(group, "layer_id", None)
            if group_id is None:
                continue
            if not getattr(group, "visible", True):
                continue
            is_active = group_id == active_group_id
            items: list[QtWidgets.QGraphicsRectItem] = []
            for row in getattr(group, "rows", []):
                rect = QtCore.QRectF(row.x1, row.y1, row.x2 - row.x1, row.y2 - row.y1).normalized()
                item = self._scene.addRect(rect)
                color = self._GROUP_COLORS.get(row.group_name, QtGui.QColor(255, 255, 255))
                pen = QtGui.QPen(color, 3 if is_active else 2)
                pen.setCosmetic(True)
                if not is_active:
                    pen.setStyle(QtCore.Qt.PenStyle.DashLine)
                item.setPen(pen)
                fill_alpha = 30 if is_active else 12
                item.setBrush(QtGui.QBrush(QtGui.QColor(color.red(), color.green(), color.blue(), fill_alpha)))
                item.setToolTip(f"{row.group_name} {row.score:.3f}")
                item.setZValue(4)
                item.setOpacity(1.0 if is_active else 0.7)
                items.append(item)
            self._group_boxes[group_id] = items

    def add_group_mask(self, group_id: str, mask_item: MaskPixmapItem) -> None:
        self._scene.addItem(mask_item)
        self._group_masks.setdefault(group_id, []).append(mask_item)

    def set_group_visible(self, group_id: str, visible: bool) -> None:
        for item in self._group_boxes.get(group_id, []):
            item.setVisible(visible)
        for item in self._group_masks.get(group_id, []):
            item.setVisible(visible)

    def set_group_opacity(self, group_id: str, opacity: float) -> None:
        for item in self._group_masks.get(group_id, []):
            item.setOpacity(float(opacity))

    def clear_group(self, group_id: str) -> None:
        self._clear_group_boxes(group_id)
        for item in self._group_masks.pop(group_id, []):
            self._scene.removeItem(item)

    def render_overlay(self) -> QtGui.QImage:
        image = QtGui.QImage(int(self.image_rect().width()), int(self.image_rect().height()), QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(image)
        self._scene.render(painter, QtCore.QRectF(image.rect()), self.image_rect())
        painter.end()
        return image

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        modifiers = event.modifiers()
        zoom_mod = bool(modifiers & (QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.MetaModifier))
        pixel_delta = event.pixelDelta()
        is_trackpad_scroll = not pixel_delta.isNull()
        if is_trackpad_scroll and not zoom_mod:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y() or event.angleDelta().x()
        if delta == 0:
            super().wheelEvent(event)
            return
        self._zoom_by(1.0015 ** delta, event.position())
        event.accept()

    def _zoom_by(self, factor: float, view_pos: QtCore.QPointF | None = None) -> None:
        factor = max(0.2, min(5.0, float(factor)))
        current = self.transform().m11()
        target = current * factor
        if target < 0.02:
            factor = 0.02 / current
        elif target > 60.0:
            factor = 60.0 / current
        if view_pos is not None:
            old_scene = self.mapToScene(view_pos.toPoint())
        self.scale(factor, factor)
        if view_pos is not None:
            new_view = self.mapFromScene(old_scene)
            delta = new_view - view_pos.toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + int(delta.y()))
        self.zoomChanged.emit(self.current_zoom())

    def viewportEvent(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.NativeGesture and isinstance(event, QtGui.QNativeGestureEvent):
            if event.gestureType() == QtCore.Qt.NativeGestureType.ZoomNativeGesture:
                self._zoom_by(1.0 + float(event.value()), event.position())
                return True
            if event.gestureType() == QtCore.Qt.NativeGestureType.SmartZoomNativeGesture:
                self.fit_image()
                return True
        return super().viewportEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        # Space → temporary pan (like Photoshop)
        if event.key() == QtCore.Qt.Key.Key_Space and not event.isAutoRepeat() and not self._space_panning:
            self._space_panning = True
            self._prev_tool = self._active_tool
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
            event.accept()
            return

        if self._active_tool is not None and self._active_tool.key_press(event):
            event.accept()
            return
        if event.key() in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            if self.delete_selected_rois():
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Space and not event.isAutoRepeat() and self._space_panning:
            self._space_panning = False
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
            if self._prev_tool is not None:
                self._prev_tool.activate(self)
            self._prev_tool = None
            event.accept()
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        scene_pos = self.mapToScene(event.position().toPoint())
        if not self._space_panning and self._active_tool is not None and self._active_tool.mouse_press(event, scene_pos):
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        scene_pos = self.mapToScene(event.position().toPoint())
        self.cursorMoved.emit(scene_pos.x(), scene_pos.y())
        if not self._space_panning and self._active_tool is not None and self._active_tool.mouse_move(event, scene_pos):
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        scene_pos = self.mapToScene(event.position().toPoint())
        if not self._space_panning and self._active_tool is not None and self._active_tool.mouse_release(event, scene_pos):
            event.accept()
            return
        super().mouseReleaseEvent(event)
