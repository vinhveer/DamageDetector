import os

from ui.qt import QtGui, QtWidgets
from ui.widgets import MaskPaintView, RoiSelectView


class RoiSelectionDialog(QtWidgets.QDialog):
    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select ROI")
        self.roi_box = None

        pixmap = QtGui.QPixmap(image_path)
        if pixmap.isNull():
            raise RuntimeError(f"Failed to load image: {image_path}")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Drag to select ROI. Ctrl+Wheel = zoom."))

        self.view = RoiSelectView(pixmap)
        self.view.setMinimumHeight(480)
        layout.addWidget(self.view, 1)

        self.info = QtWidgets.QLabel("ROI: (full image)")
        layout.addWidget(self.info)

        btn_row = QtWidgets.QHBoxLayout()
        layout.addLayout(btn_row)
        self.full_btn = QtWidgets.QPushButton("Use full image")
        self.ok_btn = QtWidgets.QPushButton("OK")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_row.addWidget(self.full_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.ok_btn)
        btn_row.addWidget(self.cancel_btn)

        self.view.roi_changed.connect(self._on_roi_changed)
        self.full_btn.clicked.connect(self._use_full)
        self.ok_btn.clicked.connect(self._accept_roi)
        self.cancel_btn.clicked.connect(self.reject)

        self.resize(900, 700)

    def _on_roi_changed(self, roi_box):
        if roi_box is None:
            self.info.setText("ROI: (full image)")
        else:
            l, t, r, b = roi_box
            self.info.setText(f"ROI: left={l}, top={t}, right={r}, bottom={b}  (w={r-l}, h={b-t})")

    def _use_full(self):
        self.roi_box = None
        self.accept()

    def _accept_roi(self):
        roi = self.view.selected_roi_box()
        self.roi_box = roi  # None means full image
        self.accept()


class GroundTruthEditorDialog(QtWidgets.QDialog):
    def __init__(self, image_path: str, initial_mask_path: str | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Ground Truth")
        self._image_path = image_path
        self.last_saved_path = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Paint: LMB. Erase: Ctrl+LMB. Zoom: Ctrl+Wheel."))

        self.view = MaskPaintView(image_path, initial_mask_path=initial_mask_path)
        self.view.setMinimumHeight(520)
        layout.addWidget(self.view, 1)

        tools = QtWidgets.QHBoxLayout()
        layout.addLayout(tools)
        tools.addWidget(QtWidgets.QLabel("Brush"))
        self.brush_spin = QtWidgets.QSpinBox()
        self.brush_spin.setRange(1, 200)
        self.brush_spin.setValue(self.view.brush_size)
        tools.addWidget(self.brush_spin)
        tools.addStretch(1)
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.save_btn = QtWidgets.QPushButton("Save...")
        self.close_btn = QtWidgets.QPushButton("Close")
        tools.addWidget(self.clear_btn)
        tools.addWidget(self.save_btn)
        tools.addWidget(self.close_btn)

        self.brush_spin.valueChanged.connect(self.view.set_brush_size)
        self.clear_btn.clicked.connect(self.view.clear_mask)
        self.save_btn.clicked.connect(self._save)
        self.close_btn.clicked.connect(self.accept)

        self.resize(1000, 800)

    def _save(self):
        base = os.path.splitext(os.path.basename(self._image_path))[0]
        default_name = f"{base}_gt.png"
        f, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save ground truth mask",
            default_name,
            filter="PNG (*.png);;All files (*)",
        )
        if not f:
            return
        ok = self.view.save_mask(f)
        if not ok:
            QtWidgets.QMessageBox.critical(self, "Save", f"Failed to save:\n{f}")
        else:
            self.last_saved_path = f
            QtWidgets.QMessageBox.information(self, "Save", f"Saved:\n{f}")
