from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets


class MaskPixmapItem(QtWidgets.QGraphicsPixmapItem):
    """Segmentation mask overlay on the canvas.

    Loads a grayscale PNG mask and renders it as a semi-transparent colored
    overlay positioned at *bbox* in scene (image) coordinates.
    """

    def __init__(
        self,
        mask_path: str | Path,
        bbox: QtCore.QRectF,
        color: QtGui.QColor | None = None,
        opacity: float = 0.45,
        parent: QtWidgets.QGraphicsItem | None = None,
    ) -> None:
        super().__init__(parent)
        self._mask_path = Path(mask_path)
        self._bbox = QtCore.QRectF(bbox)
        self._color = color or QtGui.QColor(55, 150, 255)

        pixmap = self._load_colored_mask()
        self.setPixmap(pixmap)
        self.setPos(bbox.topLeft())
        self.setOpacity(opacity)
        self.setZValue(5)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)

    def _load_colored_mask(self) -> QtGui.QPixmap:
        gray = QtGui.QImage(str(self._mask_path))
        if gray.isNull():
            img = QtGui.QImage(1, 1, QtGui.QImage.Format.Format_ARGB32)
            img.fill(QtCore.Qt.GlobalColor.transparent)
            return QtGui.QPixmap.fromImage(img)

        gray = gray.convertToFormat(QtGui.QImage.Format.Format_Grayscale8)
        w, h = gray.width(), gray.height()

        colored = QtGui.QImage(w, h, QtGui.QImage.Format.Format_ARGB32)
        colored.fill(QtCore.Qt.GlobalColor.transparent)

        mask_bits = gray.constBits()
        line_stride = gray.bytesPerLine()
        color = QtGui.qRgba(self._color.red(), self._color.green(), self._color.blue(), 255)
        transparent = QtGui.qRgba(0, 0, 0, 0)
        for y in range(h):
            row_offset = y * line_stride
            for x in range(w):
                colored.setPixel(x, y, color if mask_bits[row_offset + x] > 0 else transparent)

        return QtGui.QPixmap.fromImage(colored)

    @classmethod
    def from_full_image_mask(
        cls,
        mask_path: str | Path,
        image_rect: QtCore.QRectF,
        color: QtGui.QColor | None = None,
        opacity: float = 0.45,
    ) -> "MaskPixmapItem":
        """Create mask item that covers the whole image (for UNet full-image masks)."""
        return cls(mask_path, image_rect, color=color, opacity=opacity)
