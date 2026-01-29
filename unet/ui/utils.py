import numpy as np

from ui.qt import QtGui


def _qimage_to_gray_np(qimg: QtGui.QImage) -> np.ndarray:
    qimg = qimg.convertToFormat(QtGui.QImage.Format_Grayscale8)
    w = qimg.width()
    h = qimg.height()
    bpl = qimg.bytesPerLine()
    expected = h * bpl
    ptr = qimg.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8)
    if arr.size < expected:
        arr = np.frombuffer(ptr.tobytes(), dtype=np.uint8)
    arr = arr[:expected].reshape((h, bpl))[:, :w]
    return arr.copy()


def _rgb_to_qimage(rgb: np.ndarray) -> QtGui.QImage:
    h, w, _ = rgb.shape
    qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return qimg.copy()


def _gray_to_qimage(gray: np.ndarray) -> QtGui.QImage:
    h, w = gray.shape
    qimg = QtGui.QImage(gray.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    return qimg.copy()
