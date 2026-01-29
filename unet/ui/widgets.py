import math

import numpy as np

from ui.qt import QtCore, QtGui, QtWidgets


class CtrlZoomGraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )

    def wheelEvent(self, event: QtGui.QWheelEvent):
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            factor = 1.25 if delta > 0 else 0.8
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)


class RoiSelectView(CtrlZoomGraphicsView):
    roi_changed = QtCore.Signal(object)

    def __init__(self, pixmap: QtGui.QPixmap, parent=None):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setPos(0, 0)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())

        self._dragging = False
        self._start = None
        self._rect_item = None

    def _clamp_to_scene(self, p: QtCore.QPointF) -> QtCore.QPointF:
        r = self.sceneRect()
        x = max(r.left(), min(p.x(), r.right()))
        y = max(r.top(), min(p.y(), r.bottom()))
        return QtCore.QPointF(x, y)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._dragging = True
            self._start = self._clamp_to_scene(self.mapToScene(event.pos()))
            if self._rect_item is None:
                pen = QtGui.QPen(QtGui.QColor(0, 255, 0))
                pen.setStyle(QtCore.Qt.PenStyle.DashLine)
                pen.setWidth(2)
                self._rect_item = self._scene.addRect(QtCore.QRectF(), pen)
            self._rect_item.setRect(QtCore.QRectF(self._start, self._start))
            self.roi_changed.emit(self.selected_roi_box())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._dragging and self._start is not None and self._rect_item is not None:
            cur = self._clamp_to_scene(self.mapToScene(event.pos()))
            rect = QtCore.QRectF(self._start, cur).normalized()
            self._rect_item.setRect(rect)
            self.roi_changed.emit(self.selected_roi_box())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._dragging = False
            self.roi_changed.emit(self.selected_roi_box())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def selected_roi_box(self):
        if self._rect_item is None:
            return None
        rect = self._rect_item.rect().normalized()
        if rect.width() < 1 or rect.height() < 1:
            return None

        left = int(max(0, math.floor(rect.left())))
        top = int(max(0, math.floor(rect.top())))
        right = int(min(self._pixmap_item.pixmap().width(), math.ceil(rect.right())))
        bottom = int(min(self._pixmap_item.pixmap().height(), math.ceil(rect.bottom())))
        if right <= left or bottom <= top:
            return None
        return (left, top, right, bottom)


class MaskPaintView(CtrlZoomGraphicsView):
    def __init__(self, image_path: str, initial_mask_path: str | None = None, parent=None):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)

        pixmap = QtGui.QPixmap(image_path)
        if pixmap.isNull():
            raise RuntimeError(f"Failed to load image: {image_path}")

        self._img_w = pixmap.width()
        self._img_h = pixmap.height()

        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setPos(0, 0)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())

        self._overlay_item = self._scene.addPixmap(QtGui.QPixmap())
        self._overlay_item.setPos(0, 0)

        self.brush_size = 12
        self.alpha = 120

        self.mask = np.zeros((self._img_h, self._img_w), dtype=np.uint8)
        if initial_mask_path:
            self._load_initial_mask(initial_mask_path)

        self._overlay_rgba = np.zeros((self._img_h, self._img_w, 4), dtype=np.uint8)
        self._overlay_rgba[..., 0] = 255
        self._overlay_rgba[..., 1] = 0
        self._overlay_rgba[..., 2] = 0
        self._overlay_rgba[..., 3] = 0

        self._painting = False
        self._last_pt = None

        self._refresh_overlay_full()

    def _load_initial_mask(self, mask_path: str):
        mask_img = QtGui.QImage(mask_path)
        if mask_img.isNull():
            return
        mask_img = mask_img.convertToFormat(QtGui.QImage.Format_Grayscale8)
        if mask_img.width() != self._img_w or mask_img.height() != self._img_h:
            mask_img = mask_img.scaled(
                self._img_w,
                self._img_h,
                QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                QtCore.Qt.TransformationMode.FastTransformation,
            )
        h = mask_img.height()
        bpl = mask_img.bytesPerLine()
        expected = h * bpl
        ptr = mask_img.constBits()  # PySide6 returns a memoryview (no setsize()).
        arr = np.frombuffer(ptr, dtype=np.uint8)
        if arr.size < expected:
            arr = np.frombuffer(ptr.tobytes(), dtype=np.uint8)
        arr = arr[:expected].reshape((h, bpl))[:, : self._img_w]
        self.mask = (arr > 127).astype(np.uint8) * 255

    def set_brush_size(self, size: int):
        self.brush_size = max(1, int(size))

    def _scene_to_img(self, p: QtCore.QPointF):
        x = int(round(p.x()))
        y = int(round(p.y()))
        if x < 0 or y < 0 or x >= self._img_w or y >= self._img_h:
            return None
        return (x, y)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._painting = True
            p = self._scene_to_img(self.mapToScene(event.pos()))
            if p is not None:
                self._last_pt = p
                self._paint_at(p, erase=bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._painting:
            p = self._scene_to_img(self.mapToScene(event.pos()))
            if p is not None and self._last_pt is not None:
                erase = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)
                self._paint_line(self._last_pt, p, erase=erase)
                self._last_pt = p
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._painting = False
            self._last_pt = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _paint_line(self, a, b, *, erase: bool):
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay
        dist = math.hypot(dx, dy)
        step = max(1.0, self.brush_size / 2.0)
        n = max(1, int(dist / step))
        for i in range(n + 1):
            t = i / n
            x = int(round(ax + dx * t))
            y = int(round(ay + dy * t))
            self._paint_at((x, y), erase=erase, refresh=False)
        self._refresh_overlay_full()

    def _paint_at(self, p, *, erase: bool, refresh: bool = True):
        x, y = p
        r = max(1, self.brush_size // 2)
        x0 = max(0, x - r)
        x1 = min(self._img_w, x + r + 1)
        y0 = max(0, y - r)
        y1 = min(self._img_h, y + r + 1)

        yy, xx = np.ogrid[y0:y1, x0:x1]
        circle = (xx - x) * (xx - x) + (yy - y) * (yy - y) <= r * r

        if erase:
            self.mask[y0:y1, x0:x1][circle] = 0
        else:
            self.mask[y0:y1, x0:x1][circle] = 255

        if refresh:
            self._refresh_overlay_full()

    def _refresh_overlay_full(self):
        self._overlay_rgba[..., 3] = (self.mask > 0).astype(np.uint8) * self.alpha
        qimg = QtGui.QImage(
            self._overlay_rgba.data,
            self._img_w,
            self._img_h,
            4 * self._img_w,
            QtGui.QImage.Format_RGBA8888,
        )
        self._overlay_item.setPixmap(QtGui.QPixmap.fromImage(qimg.copy()))

    def clear_mask(self):
        self.mask.fill(0)
        self._refresh_overlay_full()

    def save_mask(self, path: str) -> bool:
        qimg = QtGui.QImage(
            self.mask.data,
            self._img_w,
            self._img_h,
            self._img_w,
            QtGui.QImage.Format_Grayscale8,
        ).copy()
        return qimg.save(path)
