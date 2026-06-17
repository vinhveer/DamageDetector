from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ui.core.settings import UiSettings


def _make_form() -> QtWidgets.QFormLayout:
    form = QtWidgets.QFormLayout()
    form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
    form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
    form.setVerticalSpacing(10)
    form.setHorizontalSpacing(10)
    form.setContentsMargins(16, 16, 16, 8)
    return form


def _path_row(parent: QtWidgets.QWidget, placeholder: str, ext_filter: str) -> tuple[QtWidgets.QWidget, QtWidgets.QLineEdit]:
    container = QtWidgets.QWidget(parent)
    rl = QtWidgets.QHBoxLayout(container)
    rl.setContentsMargins(0, 0, 0, 0)
    rl.setSpacing(4)
    edit = QtWidgets.QLineEdit(container)
    edit.setPlaceholderText(placeholder)
    btn = QtWidgets.QToolButton(container)
    btn.setText("…")
    btn.setFixedWidth(26)

    def _pick() -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent, "Select file", edit.text() or "", ext_filter
        )
        if path:
            edit.setText(path)

    btn.clicked.connect(_pick)
    rl.addWidget(edit)
    rl.addWidget(btn)
    return container, edit


class _ModelsTab(QtWidgets.QWidget):
    def __init__(self, settings: UiSettings, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        form = _make_form()

        self.dino_ckpt = QtWidgets.QLineEdit(self)
        self.dino_ckpt.setPlaceholderText("HuggingFace model ID or local path")
        self.dino_ckpt.setText(settings.dino_checkpoint)
        form.addRow("DINO checkpoint", self.dino_ckpt)

        stable_widget, self.stabledino_ckpt = _path_row(self, "Path to StableDINO .pth", "Checkpoint (*.pth)")
        self.stabledino_ckpt.setText(settings.stabledino_checkpoint)
        form.addRow("StableDINO checkpoint", stable_widget)

        sam_widget, self.sam_ckpt = _path_row(self, "Path to SAM .pth", "Checkpoint (*.pth)")
        self.sam_ckpt.setText(settings.sam_checkpoint)
        form.addRow("SAM checkpoint", sam_widget)

        self.sam_type = QtWidgets.QComboBox(self)
        for t in ("auto", "vit_h", "vit_l", "vit_b"):
            self.sam_type.addItem(t)
        idx = self.sam_type.findText(settings.sam_model_type)
        if idx >= 0:
            self.sam_type.setCurrentIndex(idx)
        form.addRow("SAM model type", self.sam_type)

        lora_widget, self.sam_lora_ckpt = _path_row(self, "Path to SAM-LoRA .pth", "Checkpoint (*.pth)")
        self.sam_lora_ckpt.setText(settings.sam_lora_checkpoint)
        form.addRow("SAM-LoRA checkpoint", lora_widget)

        lora_base_widget, self.sam_lora_base_ckpt = _path_row(self, "Path to base SAM .pth", "Checkpoint (*.pth)")
        self.sam_lora_base_ckpt.setText(settings.sam_lora_base_checkpoint)
        form.addRow("SAM-LoRA base", lora_base_widget)

        unet_widget, self.unet_ckpt = _path_row(self, "Path to UNet .pth", "Checkpoint (*.pth)")
        self.unet_ckpt.setText(settings.unet_checkpoint)
        form.addRow("UNet checkpoint", unet_widget)

        self.device = QtWidgets.QComboBox(self)
        for d in ("auto", "cpu", "cuda", "mps"):
            self.device.addItem(d)
        idx = self.device.findText(settings.device)
        if idx >= 0:
            self.device.setCurrentIndex(idx)
        form.addRow("Device", self.device)

        outer.addLayout(form)
        outer.addStretch(1)

    def apply_to(self, settings: UiSettings) -> None:
        settings.dino_checkpoint = self.dino_ckpt.text().strip()
        settings.stabledino_checkpoint = self.stabledino_ckpt.text().strip()
        settings.sam_checkpoint = self.sam_ckpt.text().strip()
        settings.sam_model_type = self.sam_type.currentText()
        settings.sam_lora_checkpoint = self.sam_lora_ckpt.text().strip()
        settings.sam_lora_base_checkpoint = self.sam_lora_base_ckpt.text().strip()
        settings.unet_checkpoint = self.unet_ckpt.text().strip()
        settings.device = self.device.currentText()


class _DetectionTab(QtWidgets.QWidget):
    def __init__(self, settings: UiSettings, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        form = _make_form()

        self.box_thr = QtWidgets.QDoubleSpinBox(self)
        self.box_thr.setRange(0.001, 1.0)
        self.box_thr.setDecimals(3)
        self.box_thr.setSingleStep(0.01)
        self.box_thr.setValue(settings.box_threshold)
        form.addRow("Box threshold", self.box_thr)

        self.text_thr = QtWidgets.QDoubleSpinBox(self)
        self.text_thr.setRange(0.001, 1.0)
        self.text_thr.setDecimals(3)
        self.text_thr.setSingleStep(0.01)
        self.text_thr.setValue(settings.text_threshold)
        form.addRow("Text threshold", self.text_thr)

        self.max_dets = QtWidgets.QSpinBox(self)
        self.max_dets.setRange(1, 500)
        self.max_dets.setValue(settings.max_dets)
        form.addRow("Max detections", self.max_dets)

        yolo_widget, self.yolo_ckpt = _path_row(self, "Path to YOLO best.pt", "Checkpoint (*.pt)")
        self.yolo_ckpt.setText(settings.yolo_checkpoint)
        form.addRow("YOLO checkpoint", yolo_widget)

        self.yolo_conf = QtWidgets.QDoubleSpinBox(self)
        self.yolo_conf.setRange(0.001, 1.0)
        self.yolo_conf.setDecimals(3)
        self.yolo_conf.setSingleStep(0.01)
        self.yolo_conf.setValue(settings.yolo_conf)
        form.addRow("YOLO confidence", self.yolo_conf)

        self.min_box_px = QtWidgets.QSpinBox(self)
        self.min_box_px.setRange(0, 100)
        self.min_box_px.setValue(settings.min_box_px)
        form.addRow("Min box size (px)", self.min_box_px)

        outer.addLayout(form)
        outer.addStretch(1)

    def apply_to(self, settings: UiSettings) -> None:
        settings.box_threshold = self.box_thr.value()
        settings.text_threshold = self.text_thr.value()
        settings.max_dets = self.max_dets.value()
        settings.yolo_checkpoint = self.yolo_ckpt.text().strip()
        settings.yolo_conf = float(self.yolo_conf.value())
        settings.min_box_px = self.min_box_px.value()


class _SegmentationTab(QtWidgets.QWidget):
    def __init__(self, settings: UiSettings, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        form = _make_form()

        self.unet_thr = QtWidgets.QDoubleSpinBox(self)
        self.unet_thr.setRange(0.0, 1.0)
        self.unet_thr.setDecimals(3)
        self.unet_thr.setSingleStep(0.01)
        self.unet_thr.setValue(settings.unet_threshold)
        form.addRow("UNet threshold", self.unet_thr)

        self.min_mask_area = QtWidgets.QSpinBox(self)
        self.min_mask_area.setRange(0, 50000)
        self.min_mask_area.setSingleStep(10)
        self.min_mask_area.setValue(settings.min_mask_area)
        form.addRow("Min mask area (px²)", self.min_mask_area)

        outer.addLayout(form)
        outer.addStretch(1)

    def apply_to(self, settings: UiSettings) -> None:
        settings.unet_threshold = self.unet_thr.value()
        settings.min_mask_area = self.min_mask_area.value()


class PreferencesDialog(QtWidgets.QDialog):
    """Preferences dialog: Models / Detection / Segmentation tabs."""

    def __init__(self, settings: UiSettings, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self.setWindowTitle("Preferences")
        self.resize(520, 360)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 12)
        root.setSpacing(0)

        self._tabs = QtWidgets.QTabWidget(self)
        self._tabs.setDocumentMode(True)
        self._models_tab = _ModelsTab(settings, self._tabs)
        self._detect_tab = _DetectionTab(settings, self._tabs)
        self._seg_tab = _SegmentationTab(settings, self._tabs)
        self._tabs.addTab(self._models_tab, "Models")
        self._tabs.addTab(self._detect_tab, "Detection")
        self._tabs.addTab(self._seg_tab, "Segmentation")
        root.addWidget(self._tabs, 1)

        sep = QtWidgets.QFrame(self)
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet("color: rgba(255,255,255,0.08);")
        root.addWidget(sep)

        btn_wrap = QtWidgets.QWidget(self)
        bwl = QtWidgets.QHBoxLayout(btn_wrap)
        bwl.setContentsMargins(12, 8, 12, 0)
        bwl.addStretch(1)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.reject)
        bwl.addWidget(buttons)
        root.addWidget(btn_wrap)

    def _apply(self) -> None:
        self._models_tab.apply_to(self._settings)
        self._detect_tab.apply_to(self._settings)
        self._seg_tab.apply_to(self._settings)
        self.accept()
