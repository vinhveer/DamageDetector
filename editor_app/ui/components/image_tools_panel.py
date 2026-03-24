from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from editor_app.canvas import ImageCanvas


class ImageToolsPanel(QtWidgets.QWidget):
    def __init__(self, overlay_canvas: ImageCanvas, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._overlay_canvas = overlay_canvas

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(QtWidgets.QLabel("Brush radius:", self))
        brush_row = QtWidgets.QHBoxLayout()
        self._brush_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self._brush_slider.setRange(1, 250)
        self._brush_slider.setValue(self._overlay_canvas.canvas_state().brush_radius)
        self._brush_spin = QtWidgets.QSpinBox(self)
        self._brush_spin.setRange(1, 250)
        self._brush_spin.setValue(self._brush_slider.value())
        self._brush_spin.setSuffix(" px")
        self._brush_value = QtWidgets.QLabel(f"{self._brush_slider.value()} px", self)

        def set_brush(value: int) -> None:
            self._overlay_canvas.set_brush_radius(int(value))
            self._brush_value.setText(f"{int(value)} px")

        self._brush_slider.valueChanged.connect(self._brush_spin.setValue)
        self._brush_slider.valueChanged.connect(set_brush)
        self._brush_spin.valueChanged.connect(self._brush_slider.setValue)
        self._brush_spin.valueChanged.connect(set_brush)

        brush_row.addWidget(self._brush_slider, 1)
        brush_row.addWidget(self._brush_spin)
        layout.addLayout(brush_row)
        layout.addWidget(self._brush_value)

        layout.addSpacing(6)
        layout.addWidget(QtWidgets.QLabel("Overlay opacity:", self))
        opacity_row = QtWidgets.QHBoxLayout()
        self._overlay_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self._overlay_slider.setRange(0, 255)
        self._overlay_slider.setValue(self._overlay_canvas.canvas_state().overlay_opacity)
        self._overlay_spin = QtWidgets.QSpinBox(self)
        self._overlay_spin.setRange(0, 255)
        self._overlay_spin.setValue(self._overlay_slider.value())

        def set_opacity(value: int) -> None:
            self._overlay_canvas.set_overlay_opacity(int(value))

        self._overlay_slider.valueChanged.connect(self._overlay_spin.setValue)
        self._overlay_slider.valueChanged.connect(set_opacity)
        self._overlay_spin.valueChanged.connect(self._overlay_slider.setValue)
        self._overlay_spin.valueChanged.connect(set_opacity)

        opacity_row.addWidget(self._overlay_slider, 1)
        opacity_row.addWidget(self._overlay_spin)
        layout.addLayout(opacity_row)

        layout.addSpacing(6)
        layout.addWidget(
            QtWidgets.QLabel(
                "Paint: LMB\nErase: Ctrl + LMB\nZoom: Ctrl + Wheel\nBrush: Ctrl + Shift + Wheel\nPan: Wheel",
                self,
            )
        )
        layout.addStretch(1)
