from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class SettingsWorkspace(QtWidgets.QWidget):
    settingsSaved = QtCore.Signal(dict)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._fields: dict[str, QtWidgets.QWidget] = {}
        self._dirty = False

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Default Prediction Settings", self)
        font = title.font()
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        title.setFont(font)
        header.addWidget(title)
        header.addStretch(1)
        self._status = QtWidgets.QLabel("Saved", self)
        header.addWidget(self._status)
        self._save_btn = QtWidgets.QPushButton("Save Settings", self)
        self._save_btn.clicked.connect(self._save_settings)
        self._save_btn.setEnabled(False)
        header.addWidget(self._save_btn)
        root.addLayout(header)

        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        root.addWidget(scroll, 1)

        host = QtWidgets.QWidget(scroll)
        scroll.setWidget(host)
        form_root = QtWidgets.QVBoxLayout(host)
        form_root.setContentsMargins(0, 0, 0, 0)
        form_root.setSpacing(18)

        form_root.addWidget(self._build_general_group(host))
        form_root.addWidget(self._build_sam_group(host))
        form_root.addWidget(self._build_dino_group(host))
        form_root.addWidget(self._build_unet_group(host))
        form_root.addStretch(1)

    def set_settings(self, settings: dict) -> None:
        for key, widget in self._fields.items():
            value = settings.get(key)
            if isinstance(widget, QtWidgets.QComboBox):
                widget.blockSignals(True)
                index = widget.findData(value)
                if index < 0:
                    index = widget.findText(str(value))
                if index >= 0:
                    widget.setCurrentIndex(index)
                widget.blockSignals(False)
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.blockSignals(True)
                widget.setChecked(bool(value))
                widget.blockSignals(False)
            elif isinstance(widget, QtWidgets.QSpinBox):
                widget.blockSignals(True)
                widget.setValue(int(value or 0))
                widget.blockSignals(False)
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                widget.blockSignals(True)
                widget.setValue(float(value or 0.0))
                widget.blockSignals(False)
            elif isinstance(widget, QtWidgets.QLineEdit):
                widget.blockSignals(True)
                widget.setText("" if value is None else str(value))
                widget.blockSignals(False)
        self._set_dirty(False)

    def _build_general_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("General", parent)
        form = self._section_layout(group)
        self._add_row(form, 0, "Device", self._combo("device", [("Auto", "auto"), ("CUDA", "cuda"), ("MPS", "mps"), ("CPU", "cpu")]))
        self._add_row(form, 1, "Invert mask", self._checkbox("invert_mask"))
        self._add_row(form, 2, "Min area", self._spin("min_area", maximum=1000000))
        self._add_row(form, 3, "Dilate", self._spin("dilate", maximum=4096))
        return group

    def _build_sam_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("SAM / Finetune", parent)
        form = self._section_layout(group)
        self._add_row(form, 0, "SAM checkpoint", self._line("sam_checkpoint"))
        self._add_row(form, 1, "SAM model type", self._combo("sam_model_type", [("Auto", "auto"), ("vit_b", "vit_b"), ("vit_l", "vit_l"), ("vit_h", "vit_h")]))
        self._add_row(form, 2, "Delta type", self._combo("delta_type", [("LoRA", "lora"), ("Adapter", "adapter"), ("Auto", "auto")]))
        self._add_row(form, 3, "Delta checkpoint", self._line("delta_checkpoint"))
        self._add_row(form, 4, "Middle dim", self._spin("middle_dim", minimum=1, maximum=4096))
        self._add_row(form, 5, "Scaling factor", self._double_spin("scaling_factor", minimum=0.0, maximum=16.0, step=0.05, decimals=3))
        self._add_row(form, 6, "Rank", self._spin("rank", minimum=1, maximum=512))
        return group

    def _build_dino_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("DINO", parent)
        form = self._section_layout(group)
        self._add_row(form, 0, "DINO checkpoint", self._line("dino_checkpoint"))
        self._add_row(form, 1, "Config ID", self._line("dino_config_id"))
        self._add_row(form, 2, "Text queries", self._line("text_queries"))
        self._add_row(form, 3, "Box threshold", self._double_spin("box_threshold", minimum=0.0, maximum=1.0, step=0.01))
        self._add_row(form, 4, "Text threshold", self._double_spin("text_threshold", minimum=0.0, maximum=1.0, step=0.01))
        self._add_row(form, 5, "Max detections", self._spin("max_dets", minimum=1, maximum=10000))
        return group

    def _build_unet_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("UNet", parent)
        form = self._section_layout(group)
        self._add_row(form, 0, "UNet model", self._line("unet_model"))
        self._add_row(form, 1, "Threshold", self._double_spin("unet_threshold", minimum=0.0, maximum=1.0, step=0.01))
        self._add_row(form, 2, "Postprocess", self._checkbox("unet_post"))
        self._add_row(form, 3, "Mode", self._combo("unet_mode", [("tile", "tile"), ("full", "full")]))
        self._add_row(form, 4, "Input size", self._spin("unet_input_size", minimum=32, maximum=4096))
        self._add_row(form, 5, "Overlap", self._spin("unet_overlap", minimum=0, maximum=4096))
        self._add_row(form, 6, "Tile batch", self._spin("unet_tile_batch", minimum=1, maximum=512))
        return group

    def _section_layout(self, parent: QtWidgets.QWidget) -> QtWidgets.QGridLayout:
        layout = QtWidgets.QGridLayout(parent)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setHorizontalSpacing(24)
        layout.setVerticalSpacing(14)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 1)
        return layout

    def _add_row(self, layout: QtWidgets.QGridLayout, row: int, label: str, widget: QtWidgets.QWidget) -> None:
        title = QtWidgets.QLabel(label, self)
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        title.setMinimumWidth(220)
        if isinstance(widget, QtWidgets.QLineEdit):
            widget.setMinimumWidth(520)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        elif isinstance(widget, QtWidgets.QComboBox):
            widget.setMinimumWidth(260)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            widget.setMinimumWidth(180)
        layout.addWidget(title, row, 0, alignment=QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        layout.addWidget(widget, row, 1)

    def _line(self, key: str) -> QtWidgets.QLineEdit:
        widget = QtWidgets.QLineEdit(self)
        self._fields[key] = widget
        widget.textChanged.connect(self._mark_dirty)
        return widget

    def _checkbox(self, key: str) -> QtWidgets.QCheckBox:
        widget = QtWidgets.QCheckBox(self)
        self._fields[key] = widget
        widget.toggled.connect(self._mark_dirty)
        return widget

    def _spin(self, key: str, *, minimum: int = 0, maximum: int = 100000) -> QtWidgets.QSpinBox:
        widget = QtWidgets.QSpinBox(self)
        widget.setRange(minimum, maximum)
        self._fields[key] = widget
        widget.valueChanged.connect(self._mark_dirty)
        return widget

    def _double_spin(
        self,
        key: str,
        *,
        minimum: float = 0.0,
        maximum: float = 100000.0,
        step: float = 0.1,
        decimals: int = 2,
    ) -> QtWidgets.QDoubleSpinBox:
        widget = QtWidgets.QDoubleSpinBox(self)
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setSingleStep(step)
        self._fields[key] = widget
        widget.valueChanged.connect(self._mark_dirty)
        return widget

    def _combo(self, key: str, items: list[tuple[str, object]]) -> QtWidgets.QComboBox:
        widget = QtWidgets.QComboBox(self)
        for label, data in items:
            widget.addItem(label, data)
        self._fields[key] = widget
        widget.currentIndexChanged.connect(self._mark_dirty)
        return widget

    def _collect_settings(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for key, widget in self._fields.items():
            if isinstance(widget, QtWidgets.QComboBox):
                payload[key] = widget.currentData()
            elif isinstance(widget, QtWidgets.QCheckBox):
                payload[key] = widget.isChecked()
            elif isinstance(widget, QtWidgets.QSpinBox):
                payload[key] = widget.value()
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                payload[key] = widget.value()
            elif isinstance(widget, QtWidgets.QLineEdit):
                payload[key] = widget.text().strip()
        return payload

    def _save_settings(self) -> None:
        self.settingsSaved.emit(self._collect_settings())
        self._set_dirty(False)

    def _mark_dirty(self, *_args) -> None:
        self._set_dirty(True)

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = bool(dirty)
        self._save_btn.setEnabled(self._dirty)
        self._status.setText("Unsaved changes" if self._dirty else "Saved")
