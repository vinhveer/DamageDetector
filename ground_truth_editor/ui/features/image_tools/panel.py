from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from canvas import ImageCanvas


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

        def _set_brush(v: int) -> None:
            self._overlay_canvas.set_brush_radius(int(v))

        self._brush_slider.valueChanged.connect(lambda v: self._brush_spin.setValue(int(v)))
        self._brush_slider.valueChanged.connect(_set_brush)
        self._brush_spin.valueChanged.connect(lambda v: self._brush_slider.setValue(int(v)))
        self._brush_spin.valueChanged.connect(_set_brush)

        brush_row.addWidget(self._brush_slider, 1)
        brush_row.addWidget(self._brush_spin)
        layout.addLayout(brush_row)
        self._brush_value = QtWidgets.QLabel(f"{self._brush_slider.value()} px", self)
        layout.addWidget(self._brush_value)

        layout.addSpacing(6)
        layout.addWidget(QtWidgets.QLabel("Overlay opacity:", self))
        op_row = QtWidgets.QHBoxLayout()
        self._overlay_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self._overlay_slider.setRange(0, 255)
        self._overlay_slider.setValue(self._overlay_canvas.canvas_state().overlay_opacity)
        self._overlay_spin = QtWidgets.QSpinBox(self)
        self._overlay_spin.setRange(0, 255)
        self._overlay_spin.setValue(self._overlay_slider.value())

        def _set_opacity(v: int) -> None:
            self._overlay_canvas.set_overlay_opacity(int(v))

        self._overlay_slider.valueChanged.connect(lambda v: self._overlay_spin.setValue(int(v)))
        self._overlay_slider.valueChanged.connect(_set_opacity)
        self._overlay_spin.valueChanged.connect(lambda v: self._overlay_slider.setValue(int(v)))
        self._overlay_spin.valueChanged.connect(_set_opacity)

        op_row.addWidget(self._overlay_slider, 1)
        op_row.addWidget(self._overlay_spin)
        layout.addLayout(op_row)

        layout.addSpacing(6)
        layout.addWidget(
            QtWidgets.QLabel(
                "Paint: LMB\n"
                "Erase: Ctrl + LMB\n"
                "Zoom: Ctrl + Wheel\n"
                "Brush: Ctrl + Shift + Wheel\n"
                "Pan: Wheel (Shift+Wheel = horizontal)",
                self,
            )
        )
        layout.addStretch(1)

    def brush_slider(self) -> QtWidgets.QSlider:
        return self._brush_slider

    def brush_spin(self) -> QtWidgets.QSpinBox:
        return self._brush_spin

    def brush_value_label(self) -> QtWidgets.QLabel:
        return self._brush_value

    def overlay_slider(self) -> QtWidgets.QSlider:
        return self._overlay_slider

    def overlay_spin(self) -> QtWidgets.QSpinBox:
        return self._overlay_spin
