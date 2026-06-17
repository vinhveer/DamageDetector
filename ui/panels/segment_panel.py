from __future__ import annotations

from PySide6 import QtCore, QtWidgets


def _make_form() -> QtWidgets.QFormLayout:
    form = QtWidgets.QFormLayout()
    form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
    form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
    form.setVerticalSpacing(6)
    form.setHorizontalSpacing(8)
    form.setContentsMargins(8, 8, 8, 8)
    return form


def _path_picker(line_edit: QtWidgets.QLineEdit, parent: QtWidgets.QWidget,
                 title: str, filter_str: str) -> QtWidgets.QHBoxLayout:
    row = QtWidgets.QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    btn = QtWidgets.QToolButton(parent)
    btn.setText("…")
    btn.setFixedWidth(26)

    def pick() -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(parent, title, line_edit.text() or "", filter_str)
        if path:
            line_edit.setText(path)

    btn.clicked.connect(pick)
    row.addWidget(line_edit)
    row.addWidget(btn)
    return row


class _SamOptions(QtWidgets.QWidget):
    """SAM zero-shot specific options."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        form = _make_form()

        self.ckpt = QtWidgets.QLineEdit(self)
        self.ckpt.setPlaceholderText("Path to SAM .pth checkpoint")
        form.addRow("Checkpoint", _path_picker(self.ckpt, self, "SAM checkpoint", "Checkpoint (*.pth)"))

        self.model_type = QtWidgets.QComboBox(self)
        self.model_type.addItems(["auto", "vit_h", "vit_l", "vit_b"])
        form.addRow("Model type", self.model_type)

        self.multimask = QtWidgets.QCheckBox("Generate multiple masks", self)
        form.addRow("", self.multimask)

        self.expand_box = QtWidgets.QSpinBox(self)
        self.expand_box.setRange(0, 200)
        self.expand_box.setValue(8)
        self.expand_box.setSuffix(" px")
        form.addRow("Expand box", self.expand_box)

        self.min_area = QtWidgets.QSpinBox(self)
        self.min_area.setRange(0, 100000)
        self.min_area.setSingleStep(50)
        self.min_area.setSuffix(" px²")
        form.addRow("Min area", self.min_area)

        self.setLayout(form)


class _SamLoraOptions(QtWidgets.QWidget):
    """SAM-LoRA finetuned options."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        form = _make_form()

        self.base_ckpt = QtWidgets.QLineEdit(self)
        self.base_ckpt.setPlaceholderText("Path to base SAM .pth")
        form.addRow("Base SAM", _path_picker(self.base_ckpt, self, "Base SAM", "Checkpoint (*.pth)"))

        self.lora_ckpt = QtWidgets.QLineEdit(self)
        self.lora_ckpt.setPlaceholderText("Path to coarse delta .pth")
        form.addRow("Coarse delta", _path_picker(self.lora_ckpt, self, "SAM-LoRA coarse", "Checkpoint (*.pth)"))

        self.lora_rank = QtWidgets.QSpinBox(self)
        self.lora_rank.setRange(1, 256)
        self.lora_rank.setValue(4)
        form.addRow("LoRA rank", self.lora_rank)

        self.predict_mode = QtWidgets.QComboBox(self)
        self.predict_mode.addItem("tile_full_box (coarse only)", "tile_full_box")
        self.predict_mode.addItem("coarse_refine", "coarse_refine")
        form.addRow("Predict mode", self.predict_mode)

        self.refine_ckpt = QtWidgets.QLineEdit(self)
        self.refine_ckpt.setPlaceholderText("Path to refine delta .pth (only for coarse_refine)")
        form.addRow("Refine delta", _path_picker(self.refine_ckpt, self, "SAM-LoRA refine", "Checkpoint (*.pth)"))

        self.refine_rank = QtWidgets.QSpinBox(self)
        self.refine_rank.setRange(1, 256)
        self.refine_rank.setValue(4)
        form.addRow("Refine rank", self.refine_rank)

        self.threshold = QtWidgets.QDoubleSpinBox(self)
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setDecimals(2)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(0.5)
        form.addRow("Threshold", self.threshold)

        self.setLayout(form)

        self.predict_mode.currentIndexChanged.connect(self._on_mode_changed)
        self._on_mode_changed(self.predict_mode.currentIndex())

    def _on_mode_changed(self, _idx: int) -> None:
        is_refine = self.predict_mode.currentData() == "coarse_refine"
        self.refine_ckpt.setEnabled(is_refine)
        self.refine_rank.setEnabled(is_refine)


class _UnetOptions(QtWidgets.QWidget):
    """UNet specific options."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        form = _make_form()

        self.ckpt = QtWidgets.QLineEdit(self)
        self.ckpt.setPlaceholderText("Path to UNet best_model.pth")
        form.addRow("Checkpoint", _path_picker(self.ckpt, self, "UNet checkpoint", "Checkpoint (*.pth)"))

        self.input_size = QtWidgets.QSpinBox(self)
        self.input_size.setRange(64, 2048)
        self.input_size.setSingleStep(32)
        self.input_size.setValue(512)
        self.input_size.setSuffix(" px")
        form.addRow("Input size", self.input_size)

        self.overlap = QtWidgets.QDoubleSpinBox(self)
        self.overlap.setRange(0.0, 0.9)
        self.overlap.setDecimals(2)
        self.overlap.setSingleStep(0.05)
        self.overlap.setValue(0.25)
        form.addRow("Tile overlap", self.overlap)

        self.threshold = QtWidgets.QDoubleSpinBox(self)
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setDecimals(2)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(0.5)
        form.addRow("Threshold", self.threshold)

        self.morph_close = QtWidgets.QSpinBox(self)
        self.morph_close.setRange(0, 21)
        self.morph_close.setSingleStep(2)
        self.morph_close.setValue(0)
        self.morph_close.setSuffix(" px")
        form.addRow("Morph close", self.morph_close)

        self.setLayout(form)


_BACKEND_BY_INDEX = ("sam", "sam_lora", "unet")
_INDEX_BY_BACKEND = {name: idx for idx, name in enumerate(_BACKEND_BY_INDEX)}


class _BackendCombo(QtWidgets.QComboBox):
    """Compat shim for legacy code that reads `panel.backend_combo.currentText() / currentData()`.

    Kept in sync with the active tab — selecting an item also switches the tab,
    and switching the tab updates the combo.
    """

    def __init__(self, panel: "SegmentPanel") -> None:
        super().__init__(panel)
        self._panel = panel
        self.addItem("SAM zero-shot", "sam")
        self.addItem("SAM-LoRA", "sam_lora")
        self.addItem("UNet", "unet")
        self.currentIndexChanged.connect(self._on_combo_changed)

    def _on_combo_changed(self, idx: int) -> None:
        if 0 <= idx < self._panel._tabs.count() and self._panel._tabs.currentIndex() != idx:
            self._panel._tabs.setCurrentIndex(idx)


class SegmentPanel(QtWidgets.QWidget):
    """Standalone segment panel — backends as TABS, with a shared common header."""

    runRequested = QtCore.Signal()
    cancelRequested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Common header — task group + opacity (shared across all backends)
        common = _make_form()

        self.task_group_combo = QtWidgets.QComboBox(self)
        self.task_group_combo.addItem("Crack only", "crack_only")
        self.task_group_combo.addItem("More damage", "more_damage")

        self.opacity = QtWidgets.QDoubleSpinBox(self)
        self.opacity.setRange(0.0, 1.0)
        self.opacity.setDecimals(2)
        self.opacity.setSingleStep(0.05)
        self.opacity.setValue(0.45)

        common.addRow("Task group", self.task_group_combo)
        common.addRow("Mask opacity", self.opacity)
        outer.addLayout(common)

        # Backend tabs — each tab is a backend with its own options
        self._tabs = QtWidgets.QTabWidget(self)
        self._tabs.setDocumentMode(True)
        self._tabs.setUsesScrollButtons(False)
        self._tabs.tabBar().setExpanding(True)

        self.sam_options = _SamOptions(self)
        self.sam_lora_options = _SamLoraOptions(self)
        self.unet_options = _UnetOptions(self)

        self._tabs.addTab(self.sam_options, "SAM zero-shot")
        self._tabs.addTab(self.sam_lora_options, "SAM-LoRA")
        self._tabs.addTab(self.unet_options, "UNet")
        outer.addWidget(self._tabs, 1)

        # Compat: keep `backend_combo` so legacy code (segment_mixin) keeps working
        self.backend_combo = _BackendCombo(self)
        self.backend_combo.hide()  # not displayed — tabs are the visible UI
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # Run / Cancel
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(8, 4, 8, 8)
        btn_row.setSpacing(6)
        self.run_button = QtWidgets.QPushButton("▶  Run Segment", self)
        self.cancel_button = QtWidgets.QPushButton("Cancel", self)
        self.cancel_button.setEnabled(False)
        self.run_button.clicked.connect(self.runRequested)
        self.cancel_button.clicked.connect(self.cancelRequested)
        btn_row.addWidget(self.run_button, 2)
        btn_row.addWidget(self.cancel_button, 1)
        outer.addLayout(btn_row)

    def _on_tab_changed(self, idx: int) -> None:
        if 0 <= idx < self.backend_combo.count() and self.backend_combo.currentIndex() != idx:
            self.backend_combo.setCurrentIndex(idx)

    def current_backend(self) -> str:
        idx = self._tabs.currentIndex()
        return _BACKEND_BY_INDEX[idx] if 0 <= idx < len(_BACKEND_BY_INDEX) else "sam"

    def set_backend(self, backend: str) -> None:
        idx = _INDEX_BY_BACKEND.get(backend)
        if idx is not None:
            self._tabs.setCurrentIndex(idx)

    @property
    def sam_ckpt(self) -> QtWidgets.QLineEdit:
        """Compat: legacy code reads `segment_panel.sam_ckpt.text()`. Returns active backend's ckpt field."""
        backend = self.current_backend()
        if backend == "sam":
            return self.sam_options.ckpt
        if backend == "sam_lora":
            return self.sam_lora_options.lora_ckpt
        return self.unet_options.ckpt
