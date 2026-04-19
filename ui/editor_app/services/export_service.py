from __future__ import annotations

from PySide6 import QtCore, QtGui


class ExportService:
    def build_overlay_image(
        self,
        *,
        base_image: QtGui.QImage,
        overlay_visual: QtGui.QImage,
        mask_image: QtGui.QImage,
        overlay_opacity: int,
    ) -> QtGui.QImage:
        if base_image.isNull():
            return QtGui.QImage()
        result = base_image.copy()
        if not overlay_visual.isNull():
            painter = QtGui.QPainter(result)
            painter.setOpacity(int(overlay_opacity) / 255.0)
            painter.drawImage(0, 0, overlay_visual)
            painter.end()
            return result
        if mask_image.isNull():
            return result
        overlay = mask_image.convertToFormat(QtGui.QImage.Format_Indexed8)
        opacity = int(overlay_opacity)
        color_table = [QtGui.QColor(255, 0, 0, (i * opacity) // 255).rgba() for i in range(256)]
        overlay.setColorTable(color_table)
        painter = QtGui.QPainter(result)
        painter.drawImage(0, 0, overlay)
        painter.end()
        return result

    def build_overlay_boxes_image(
        self,
        *,
        base_image: QtGui.QImage,
        overlay_visual: QtGui.QImage,
        mask_image: QtGui.QImage,
        overlay_opacity: int,
        detections: list[dict],
    ) -> QtGui.QImage:
        result = self.build_overlay_image(
            base_image=base_image,
            overlay_visual=overlay_visual,
            mask_image=mask_image,
            overlay_opacity=overlay_opacity,
        )
        if result.isNull():
            return result
        painter = QtGui.QPainter(result)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        for det in detections:
            box = det.get("box")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            label = str(det.get("label") or "")
            rect = QtCore.QRectF(float(box[0]), float(box[1]), float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 180), 2))
            painter.drawRect(rect)
            if label:
                painter.drawText(rect.topLeft() + QtCore.QPointF(4, 14), label)
        painter.end()
        return result
