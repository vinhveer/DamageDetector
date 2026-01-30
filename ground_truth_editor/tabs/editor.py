from PySide6 import QtCore, QtGui, QtWidgets
from canvas import ImageCanvas

class EditorTab(QtWidgets.QWidget):
    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        open_action: QtGui.QAction,
        save_action: QtGui.QAction,
        canvas: ImageCanvas,
    ) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        
        row = QtWidgets.QHBoxLayout()
        open_mask_btn = QtWidgets.QToolButton(self)
        open_mask_btn.setDefaultAction(open_action)
        save_mask_btn = QtWidgets.QToolButton(self)
        save_mask_btn.setDefaultAction(save_action)
        row.addWidget(open_mask_btn)
        row.addWidget(save_mask_btn)
        row.addStretch(1)
        layout.addLayout(row)

        layout.addWidget(QtWidgets.QLabel("Brush size:", self))
        self._brush_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self._brush_slider.setRange(1, 250)
        self._brush_slider.setValue(canvas.canvas_state().brush_radius)
        self._brush_slider.valueChanged.connect(canvas.set_brush_radius)
        layout.addWidget(self._brush_slider)
        
        self._brush_value = QtWidgets.QLabel(f"{self._brush_slider.value()} px", self)
        self._brush_slider.valueChanged.connect(lambda v: self._brush_value.setText(f"{v} px"))
        layout.addWidget(self._brush_value)

        layout.addSpacing(6)
        layout.addWidget(QtWidgets.QLabel("Overlay opacity:", self))
        self._overlay_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self._overlay_slider.setRange(0, 255)
        self._overlay_slider.setValue(canvas.canvas_state().overlay_opacity)
        self._overlay_slider.valueChanged.connect(canvas.set_overlay_opacity)
        layout.addWidget(self._overlay_slider)

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

        # Sync slider if canvas changes externally
        canvas.brushRadiusChanged.connect(self._brush_slider.setValue)
