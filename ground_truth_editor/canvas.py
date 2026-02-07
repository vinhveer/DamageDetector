from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPoint, QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QImage, QKeyEvent, QMouseEvent, QPainter, QPaintEvent, QPen, QWheelEvent
from PySide6.QtWidgets import QApplication, QWidget


@dataclass(frozen=True)
class CanvasState:
    image_loaded: bool
    mask_loaded: bool
    brush_radius: int
    overlay_opacity: int


class ImageCanvas(QWidget):
    stateChanged = Signal(CanvasState)
    cursorInfo = Signal(int, int, int)  # x, y, mask(0/1)
    brushRadiusChanged = Signal(int)
    maskChanged = Signal()
    roiSelected = Signal(object)  # roi_box (l,t,r,b) or None (full image)
    roiCanceled = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        render_mode: str = "overlay",
        editable: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self._render_mode = str(render_mode)
        self._editable = bool(editable)

        self._image = QImage()
        self._mask = QImage()  # QImage.Format_Grayscale8, 0/255
        self._overlay_cache = QImage()
        self._overlay_dirty = False

        self._overlay_opacity = 120
        self._brush_radius = 18
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)  # widget pixels

        self._is_drawing = False
        self._last_img_pt: QPoint | None = None
        self._cursor_widget_pt: QPointF | None = None

        self._roi_mode = False
        self._roi_dragging = False
        self._roi_start: QPoint | None = None
        self._roi_end: QPoint | None = None
        
        self._highlight_box: QRectF | None = None

        self._overlay_timer = QTimer(self)
        self._overlay_timer.setSingleShot(True)
        self._overlay_timer.timeout.connect(self._rebuild_overlay_cache)

        self._emit_state()

    def canvas_state(self) -> CanvasState:
        return CanvasState(
            image_loaded=not self._image.isNull(),
            mask_loaded=not self._mask.isNull(),
            brush_radius=self._brush_radius,
            overlay_opacity=self._overlay_opacity,
        )

    def set_image(self, image: QImage) -> None:
        self._image = image
        self._overlay_cache = QImage()
        self._overlay_dirty = True
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._emit_state()
        self.update()

    def set_mask(self, mask: QImage) -> None:
        self._mask = mask
        self._request_overlay_update()
        self._emit_state()
        self.update()

    def image(self) -> QImage:
        return self._image

    def mask(self) -> QImage:
        return self._mask

    def set_brush_radius(self, radius: int) -> None:
        new_radius = max(1, int(radius))
        if new_radius == self._brush_radius:
            return
        self._brush_radius = new_radius
        self.brushRadiusChanged.emit(self._brush_radius)
        self._emit_state()
        self.update()

    def set_overlay_opacity(self, value: int) -> None:
        self._overlay_opacity = max(0, min(255, int(value)))
        self._request_overlay_update()
        self._emit_state()
        self.update()

    def set_render_mode(self, mode: str) -> None:
        self._render_mode = str(mode)
        self.update()

    def set_editable(self, editable: bool) -> None:
        self._editable = bool(editable)
        self.update()

    def start_roi_selection(self) -> None:
        self._roi_mode = True
        self._roi_dragging = False
        self._roi_start = None
        self._roi_end = None
        self.setFocus(Qt.FocusReason.OtherFocusReason)
        self.update()

    def cancel_roi_selection(self) -> None:
        if not self._roi_mode:
            return
        self._roi_mode = False
        self._roi_dragging = False
        self._roi_start = None
        self._roi_end = None
        self.roiCanceled.emit()
        self.update()

    def _finish_roi_selection(self, roi_box) -> None:
        self._roi_mode = False
        self._roi_dragging = False
        self._roi_start = None
        self._roi_end = None
        self.roiSelected.emit(roi_box)
        self.update()

    def set_highlight_box(self, box: QRectF | None) -> None:
        self._highlight_box = box
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self._image.isNull() and (self._render_mode != "mask" or self._mask.isNull()):
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(self.rect(), Qt.AlignCenter, "File > Open Image...")
            return

        target = self._target_rect()
        if self._render_mode == "mask":
            if self._mask.isNull():
                painter.setPen(QColor(220, 220, 220))
                painter.drawText(self.rect(), Qt.AlignCenter, "No mask")
                return
            painter.drawImage(target, self._mask)
        else:
            painter.drawImage(target, self._image)

        if self._render_mode == "overlay" and (not self._mask.isNull()) and (not self._overlay_cache.isNull()):
            painter.drawImage(target, self._overlay_cache)

        if self._render_mode == "overlay" and self._roi_mode:
            self._draw_roi_rect(painter, target)
        elif self._render_mode == "overlay" and self._editable:
            self._draw_brush_cursor(painter, target)
            
        if self._highlight_box is not None:
            self._draw_highlight_box(painter, target)

    def _draw_highlight_box(self, painter: QPainter, target: QRectF) -> None:
        if self._image.isNull():
             return
        
        # _highlight_box is in image coordinates (x, y, w, h)
        # Convert to widget coords
        # Top-left
        tl = self._image_to_widget(self._highlight_box.topLeft(), target)
        # Bottom-right
        br = self._image_to_widget(self._highlight_box.bottomRight(), target)
        
        rect = QRectF(tl, br).normalized()
        
        # Draw box
        pen = QPen(QColor(0, 255, 255), 3, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

    def _draw_roi_rect(self, painter: QPainter, target: QRectF) -> None:
        if self._roi_start is None or self._roi_end is None or self._image.isNull():
            return

        a = self._image_to_widget(QPointF(self._roi_start), target)
        b = self._image_to_widget(QPointF(self._roi_end), target)
        rect = QRectF(a, b).normalized()
        pen = QPen(QColor(0, 255, 0), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

    def _draw_brush_cursor(self, painter: QPainter, target: QRectF) -> None:
        if self._cursor_widget_pt is None or self._image.isNull():
            return

        img_pt = self._widget_to_image(self._cursor_widget_pt, target)
        if img_pt is None:
            return

        scale = target.width() / self._image.width()
        r = self._brush_radius * scale
        center = self._image_to_widget(QPointF(img_pt.x() + 0.5, img_pt.y() + 0.5), target)

        erase = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        color = QColor(255, 80, 80) if erase else QColor(80, 255, 120)
        pen = QPen(color, 1.25)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(center, r, r)

    def _target_rect(self) -> QRectF:
        w, h = self.width(), self.height()
        if self._render_mode == "mask" and self._image.isNull() and (not self._mask.isNull()):
            iw, ih = self._mask.width(), self._mask.height()
        else:
            iw, ih = self._image.width(), self._image.height()
        if iw <= 0 or ih <= 0 or w <= 0 or h <= 0:
            return QRectF()

        fit_scale = min(w / iw, h / ih)
        scale = fit_scale * self._zoom
        tw, th = iw * scale, ih * scale
        left = (w - tw) / 2 + self._pan.x()
        top = (h - th) / 2 + self._pan.y()
        return QRectF(left, top, tw, th)

    def _widget_to_image(self, pt: QPointF, target: QRectF) -> QPoint | None:
        if target.isNull() or not target.contains(pt):
            return None
        base_w = self._mask.width() if (self._render_mode == "mask" and self._image.isNull() and (not self._mask.isNull())) else self._image.width()
        base_h = self._mask.height() if (self._render_mode == "mask" and self._image.isNull() and (not self._mask.isNull())) else self._image.height()
        if base_w <= 0 or base_h <= 0:
            return None
        scale = target.width() / base_w
        x = int((pt.x() - target.left()) / scale)
        y = int((pt.y() - target.top()) / scale)
        if x < 0 or y < 0 or x >= base_w or y >= base_h:
            return None
        return QPoint(x, y)

    def _image_to_widget(self, pt: QPointF, target: QRectF) -> QPointF:
        base_w = self._mask.width() if (self._render_mode == "mask" and self._image.isNull() and (not self._mask.isNull())) else self._image.width()
        if base_w <= 0:
            return QPointF()
        scale = target.width() / base_w
        return QPointF(target.left() + pt.x() * scale, target.top() + pt.y() * scale)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._render_mode == "overlay" and self._roi_mode and event.button() == Qt.LeftButton:
            target = self._target_rect()
            img_pt = self._widget_to_image(event.position(), target)
            if img_pt is not None:
                self._roi_dragging = True
                self._roi_start = img_pt
                self._roi_end = img_pt
                event.accept()
                self.update()
                return
        if self._editable and self._render_mode == "overlay" and (not self._roi_mode) and event.button() == Qt.LeftButton:
            self._is_drawing = True
            self._apply_brush(event.position(), event.modifiers())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._cursor_widget_pt = event.position()
        if self._render_mode == "overlay" and self._roi_mode and self._roi_dragging and (event.buttons() & Qt.LeftButton):
            target = self._target_rect()
            img_pt = self._widget_to_image(event.position(), target)
            if img_pt is not None:
                self._roi_end = img_pt
                event.accept()
                self.update()
                return
        if self._editable and self._render_mode == "overlay" and (not self._roi_mode) and self._is_drawing and (event.buttons() & Qt.LeftButton):
            self._apply_brush(event.position(), event.modifiers())
        else:
            self._emit_cursor_info(event.position())
        self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._render_mode == "overlay" and self._roi_mode and event.button() == Qt.LeftButton:
            self._roi_dragging = False
            if self._roi_start is None or self._roi_end is None:
                self._finish_roi_selection(None)
                event.accept()
                return

            x0 = min(self._roi_start.x(), self._roi_end.x())
            y0 = min(self._roi_start.y(), self._roi_end.y())
            x1 = max(self._roi_start.x(), self._roi_end.x())
            y1 = max(self._roi_start.y(), self._roi_end.y())

            if (x1 - x0) < 2 or (y1 - y0) < 2:
                self._finish_roi_selection(None)
                event.accept()
                return

            roi_box = (x0, y0, x1 + 1, y1 + 1)
            self._finish_roi_selection(roi_box)
            event.accept()
            return

        if self._editable and self._render_mode == "overlay" and (not self._roi_mode) and event.button() == Qt.LeftButton:
            self._is_drawing = False
            self._last_img_pt = None
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self._cursor_widget_pt = None
        self.update()
        super().leaveEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._roi_mode and event.key() == Qt.Key_Escape:
            self.cancel_roi_selection()
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        mods = event.modifiers()
        angle = event.angleDelta()
        pixel = event.pixelDelta()

        step_y = 0
        if angle.y():
            step_y = int(angle.y() / 120)
        elif pixel.y():
            step_y = int(round(pixel.y() / 40))

        if (mods & Qt.ControlModifier) and (mods & Qt.ShiftModifier):
            if step_y:
                self.set_brush_radius(self._brush_radius + step_y * 2)
                event.accept()
                return
            super().wheelEvent(event)
            return

        if mods & Qt.ControlModifier:
            if (self._image.isNull() and self._render_mode != "mask") or not step_y:
                super().wheelEvent(event)
                return

            target_before = self._target_rect()
            anchor_img = self._widget_to_image(event.position(), target_before)
            old_zoom = self._zoom
            factor = 1.1**step_y
            self._zoom = max(0.1, min(20.0, self._zoom * factor))

            if anchor_img is not None and self._zoom != old_zoom:
                target_after = self._target_rect()
                anchor_widget_after = self._image_to_widget(QPointF(anchor_img), target_after)
                delta = event.position() - anchor_widget_after
                self._pan = QPointF(self._pan.x() + delta.x(), self._pan.y() + delta.y())

            self.update()
            event.accept()
            return

        dx = pixel.x()
        dy = pixel.y()
        if not dx and not dy:
            dx = int(round((angle.x() / 120) * 40))
            dy = int(round((angle.y() / 120) * 40))

        if not dx and not dy:
            super().wheelEvent(event)
            return

        if (mods & Qt.ShiftModifier) and not dx:
            dx, dy = dy, 0

        self._pan = QPointF(self._pan.x() + dx, self._pan.y() + dy)
        self.update()
        event.accept()

    def _emit_cursor_info(self, widget_pos: QPointF) -> None:
        if self._image.isNull() or self._mask.isNull():
            return
        target = self._target_rect()
        img_pt = self._widget_to_image(widget_pos, target)
        if img_pt is None:
            return

        v = self._mask_value01(img_pt.x(), img_pt.y())
        self.cursorInfo.emit(img_pt.x(), img_pt.y(), v)

    def _mask_value01(self, x: int, y: int) -> int:
        if self._mask.isNull():
            return 0
        buf = self._mask.constBits()
        return 1 if buf[y * self._mask.bytesPerLine() + x] else 0

    def _apply_brush(self, widget_pos: QPointF, modifiers: Qt.KeyboardModifiers) -> None:
        if self._image.isNull() or self._mask.isNull():
            return
        target = self._target_rect()
        img_pt = self._widget_to_image(widget_pos, target)
        if img_pt is None:
            return

        erase = bool(modifiers & Qt.ControlModifier)
        value = 0 if erase else 255

        painter = QPainter(self._mask)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        pen = QPen(QColor(value, value, value), self._brush_radius * 2)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        if self._last_img_pt is None:
            painter.drawLine(img_pt, img_pt)
        else:
            painter.drawLine(self._last_img_pt, img_pt)

        painter.end()

        self._last_img_pt = img_pt
        self._request_overlay_update()
        self._emit_cursor_info(widget_pos)
        self.maskChanged.emit()
        self.update()

    def _request_overlay_update(self) -> None:
        if self._image.isNull() or self._mask.isNull():
            return
        self._overlay_dirty = True
        if not self._overlay_timer.isActive():
            self._overlay_timer.start(16)

    def _rebuild_overlay_cache(self) -> None:
        if not self._overlay_dirty or self._image.isNull() or self._mask.isNull():
            return
        self._overlay_dirty = False

        overlay = self._mask.convertToFormat(QImage.Format_Indexed8)
        color_table = []
        for i in range(256):
            a = (i * self._overlay_opacity) // 255
            color_table.append(QColor(255, 0, 0, a).rgba())
        overlay.setColorTable(color_table)
        self._overlay_cache = overlay
        self.update()

    def _emit_state(self) -> None:
        self.stateChanged.emit(self.canvas_state())
