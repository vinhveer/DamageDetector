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
        title = QtWidgets.QLabel("Prediction Settings", self)
        font = title.font()
        font.setPointSize(font.pointSize() + 2)
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

        self._tabs = QtWidgets.QTabWidget(self)
        root.addWidget(self._tabs, 1)

        self._tabs.addTab(self._build_tab(self._build_general_group), "General")
        self._tabs.addTab(self._build_tab(self._build_sam_group, self._build_sam_auto_group, self._build_sam_runtime_group), "SAM")
        self._tabs.addTab(self._build_tab(self._build_sam_lora_group), "SAM LoRA")
        self._tabs.addTab(self._build_tab(self._build_dino_group, self._build_dino_advanced_group), "DINO")
        self._tabs.addTab(self._build_tab(self._build_unet_group), "UNet")
        self._tabs.addTab(self._build_tab(self._build_crack_only_group, self._build_more_damage_group), "Tasks")
        self._tabs.addTab(self._build_tab(self._build_isolate_group), "Isolate")

    def _build_tab(self, *builders) -> QtWidgets.QWidget:
        host = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        scroll = QtWidgets.QScrollArea(host)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, 1)
        content = QtWidgets.QWidget(scroll)
        scroll.setWidget(content)
        form_root = QtWidgets.QVBoxLayout(content)
        form_root.setContentsMargins(0, 0, 0, 0)
        form_root.setSpacing(18)
        for builder in builders:
            form_root.addWidget(builder(content))
        form_root.addStretch(1)
        return host

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
                widget.setValue(int(value if value is not None else 0))
                widget.blockSignals(False)
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                widget.blockSignals(True)
                widget.setValue(float(value if value is not None else 0.0))
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
        self._add_row(form, 2, "Min area", self._spin("min_area", minimum=0, maximum=1_000_000))
        self._add_row(form, 3, "Dilate", self._spin("dilate", minimum=0, maximum=4096))
        return group

    def _build_sam_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("SAM", parent)
        form = self._section_layout(group)
        self._add_row(form, 0, "SAM checkpoint", self._line("sam_checkpoint"))
        self._add_row(form, 1, "SAM model type", self._combo("sam_model_type", [("Auto", "auto"), ("vit_b", "vit_b"), ("vit_l", "vit_l"), ("vit_h", "vit_h")]))
        return group

    def _build_sam_lora_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("SAM Finetune with LoRA", parent)
        form = self._section_layout(group)
        self._add_row(form, 0, "LoRA checkpoint", self._line("sam_lora_checkpoint"))
        self._add_row(form, 1, "Middle dim", self._spin("sam_lora_middle_dim", minimum=1, maximum=4096))
        self._add_row(form, 2, "Scaling factor", self._double_spin("sam_lora_scaling_factor", minimum=0.0, maximum=16.0, step=0.05, decimals=3))
        self._add_row(form, 3, "Rank", self._spin("sam_lora_rank", minimum=1, maximum=1024))
        return group

    def _build_dino_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("DINO", parent)
        form = self._section_layout(group)
        self._add_row(form, 0, "DINO checkpoint", self._line("dino_checkpoint"))
        self._add_row(form, 1, "Config ID", self._line("dino_config_id"))
        self._add_row(form, 2, "Box threshold", self._double_spin("box_threshold", minimum=0.0, maximum=1.0, step=0.01, decimals=4))
        self._add_row(form, 3, "Text threshold", self._double_spin("text_threshold", minimum=0.0, maximum=1.0, step=0.01, decimals=4))
        self._add_row(form, 4, "Max detections", self._spin("max_dets", minimum=1, maximum=10000))
        return group

    def _build_unet_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("UNet", parent)
        form = self._section_layout(group)
        self._add_row(form, 0, "UNet model", self._line("unet_model"))
        self._add_row(form, 1, "Threshold", self._double_spin("unet_threshold", minimum=0.0, maximum=1.0, step=0.01, decimals=4))
        self._add_row(form, 2, "Postprocess", self._checkbox("unet_post"))
        self._add_row(form, 3, "Mode", self._combo("unet_mode", [("tile", "tile"), ("letterbox", "letterbox"), ("resize", "resize")]))
        self._add_row(form, 4, "Input size", self._spin("unet_input_size", minimum=32, maximum=4096))
        self._add_row(form, 5, "Overlap", self._spin("unet_overlap", minimum=0, maximum=4096))
        self._add_row(form, 6, "Tile batch", self._spin("unet_tile_batch", minimum=1, maximum=512))
        return group

    def _build_crack_only_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Crack Only", parent)
        form = self._section_layout(group)
        self._add_row(form, 0, "Text queries", self._line("crack_text_queries"))
        return group

    def _build_more_damage_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("More Damage", parent)
        form = self._section_layout(group)
        self._add_row(form, 0, "Text queries", self._line("more_damage_text_queries"))
        self._add_row(form, 1, "Max masks", self._spin("more_damage_max_masks", minimum=1, maximum=128))
        self._add_row(
            form,
            2,
            "Crack mask model",
            self._combo(
                "more_damage_crack_mask_model",
                [
                    ("Off", "off"),
                    ("SAM Finetune with LoRA", "sam_lora"),
                    ("UNet", "unet"),
                ],
            ),
        )
        return group

    def _build_sam_auto_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("SAM Auto Mask", parent)
        form = self._section_layout(group)
        self._add_row(form, 0, "Auto profile", self._combo("sam_auto_profile", [("Auto", "auto"), ("FAST", "FAST"), ("QUALITY", "QUALITY"), ("ULTRA", "ULTRA")]))
        self._add_row(form, 1, "Points per side", self._spin("sam_points_per_side", minimum=-1, maximum=4096))
        self._add_row(form, 2, "Points per batch", self._spin("sam_points_per_batch", minimum=-1, maximum=4096))
        self._add_row(form, 3, "Pred IoU threshold", self._double_spin("sam_pred_iou_thresh", minimum=-1.0, maximum=1.0, step=0.01, decimals=4))
        self._add_row(form, 4, "Stability threshold", self._double_spin("sam_stability_score_thresh", minimum=-1.0, maximum=1.0, step=0.01, decimals=4))
        self._add_row(form, 5, "Stability offset", self._double_spin("sam_stability_score_offset", minimum=-1.0, maximum=4.0, step=0.01, decimals=4))
        self._add_row(form, 6, "Box NMS threshold", self._double_spin("sam_box_nms_thresh", minimum=-1.0, maximum=1.0, step=0.01, decimals=4))
        self._add_row(form, 7, "Crop layers", self._spin("sam_crop_n_layers", minimum=-1, maximum=16))
        self._add_row(form, 8, "Crop overlap ratio", self._double_spin("sam_crop_overlap_ratio", minimum=-1.0, maximum=1.0, step=0.01, decimals=4))
        self._add_row(form, 9, "Crop NMS threshold", self._double_spin("sam_crop_nms_thresh", minimum=-1.0, maximum=1.0, step=0.01, decimals=4))
        self._add_row(form, 10, "Crop downscale factor", self._spin("sam_crop_n_points_downscale_factor", minimum=-1, maximum=32))
        self._add_row(form, 11, "Min mask region area", self._spin("sam_min_mask_region_area", minimum=-1, maximum=1_000_000))
        return group

    def _build_sam_runtime_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("SAM Runtime", parent)
        form = self._section_layout(group)
        self._add_row(form, 0, "Seed", self._spin("sam_seed", minimum=0, maximum=1_000_000_000))
        self._add_row(form, 1, "Overlay alpha", self._double_spin("sam_overlay_alpha", minimum=0.0, maximum=1.0, step=0.05, decimals=3))
        return group

    def _build_dino_advanced_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("DINO Advanced", parent)
        form = self._section_layout(group)
        self._add_row(form, 0, "NMS IoU threshold", self._double_spin("dino_nms_iou_threshold", minimum=0.0, maximum=1.0, step=0.01, decimals=4))
        self._add_row(form, 1, "Parent contain threshold", self._double_spin("dino_parent_contain_threshold", minimum=0.0, maximum=1.0, step=0.01, decimals=4))
        self._add_row(form, 2, "Recursive min box px", self._spin("dino_recursive_min_box_px", minimum=1, maximum=4096))
        self._add_row(form, 3, "Recursive max depth", self._spin("dino_recursive_max_depth", minimum=0, maximum=16))
        self._add_row(form, 4, "Predict: tile large images", self._checkbox("predict_use_tiled_dino"))
        self._add_row(form, 5, "Predict tile trigger px", self._spin("predict_tile_trigger_px", minimum=128, maximum=16384))
        return group

    def _build_isolate_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Isolate", parent)
        form = self._section_layout(group)
        self._add_row(form, 0, "Crop to bbox", self._checkbox("isolate_crop"))
        self._add_row(form, 1, "Outside white", self._checkbox("isolate_outside_white"))
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
        title.setMinimumWidth(240)
        if isinstance(widget, QtWidgets.QLineEdit):
            widget.setMinimumWidth(520)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        elif isinstance(widget, QtWidgets.QComboBox):
            widget.setMinimumWidth(280)
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
