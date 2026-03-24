from __future__ import annotations

from PySide6 import QtWidgets


def browse_file(parent: QtWidgets.QWidget, line_edit: QtWidgets.QLineEdit, file_filter: str) -> None:
    path, _ = QtWidgets.QFileDialog.getOpenFileName(parent, "Select File", line_edit.text().strip(), file_filter)
    if path:
        line_edit.setText(path)


def set_combo_by_data(combo: QtWidgets.QComboBox, value: object) -> None:
    index = combo.findData(value)
    if index < 0:
        index = combo.findText("" if value is None else str(value))
    if index >= 0:
        combo.setCurrentIndex(index)


class SamSettingsForm(QtWidgets.QWidget):
    def __init__(self, settings: dict, widgets: dict[str, QtWidgets.QWidget], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        sam_group = QtWidgets.QGroupBox("SAM", self)
        sam_form = QtWidgets.QFormLayout(sam_group)
        sam_ckpt = QtWidgets.QLineEdit(str(settings.get("sam_checkpoint") or ""), sam_group)
        sam_btn = QtWidgets.QPushButton("Browse...", sam_group)
        sam_btn.clicked.connect(lambda: browse_file(self, sam_ckpt, "Checkpoint (*.pth);;All files (*)"))
        sam_ckpt_row = QtWidgets.QHBoxLayout()
        sam_ckpt_row.addWidget(sam_ckpt, 1)
        sam_ckpt_row.addWidget(sam_btn)
        sam_type = QtWidgets.QComboBox(sam_group)
        sam_type.addItems(["auto", "vit_b", "vit_l", "vit_h"])
        sam_type.setCurrentText(str(settings.get("sam_model_type") or "auto"))
        sam_form.addRow("Checkpoint", self._wrap_row(sam_ckpt_row))
        sam_form.addRow("Model type", sam_type)
        root.addWidget(sam_group)

        delta_group = QtWidgets.QGroupBox("Fine-tune", self)
        delta_form = QtWidgets.QFormLayout(delta_group)
        delta_type = QtWidgets.QComboBox(delta_group)
        delta_type.addItems(["adapter", "lora", "both", "auto"])
        delta_type.setCurrentText(str(settings.get("delta_type") or "lora"))
        delta_ckpt = QtWidgets.QLineEdit(str(settings.get("delta_checkpoint") or "auto"), delta_group)
        delta_btn = QtWidgets.QPushButton("Browse...", delta_group)
        delta_btn.clicked.connect(lambda: browse_file(self, delta_ckpt, "Checkpoint (*.pth);;All files (*)"))
        delta_ckpt_row = QtWidgets.QHBoxLayout()
        delta_ckpt_row.addWidget(delta_ckpt, 1)
        delta_ckpt_row.addWidget(delta_btn)
        middle_dim = QtWidgets.QSpinBox(delta_group)
        middle_dim.setRange(1, 4096)
        middle_dim.setValue(int(settings.get("middle_dim", 32)))
        scaling = QtWidgets.QDoubleSpinBox(delta_group)
        scaling.setRange(0.0, 10.0)
        scaling.setDecimals(4)
        scaling.setSingleStep(0.05)
        scaling.setValue(float(settings.get("scaling_factor", 0.2)))
        rank = QtWidgets.QSpinBox(delta_group)
        rank.setRange(1, 1024)
        rank.setValue(int(settings.get("rank", 4)))
        delta_form.addRow("Delta type", delta_type)
        delta_form.addRow("Delta checkpoint", self._wrap_row(delta_ckpt_row))
        delta_form.addRow("Middle dim", middle_dim)
        delta_form.addRow("Scaling factor", scaling)
        delta_form.addRow("Rank", rank)
        root.addWidget(delta_group)

        mask_group = QtWidgets.QGroupBox("Mask postprocess", self)
        mask_form = QtWidgets.QFormLayout(mask_group)
        invert = QtWidgets.QCheckBox(mask_group)
        invert.setChecked(bool(settings.get("invert_mask", False)))
        min_area = QtWidgets.QSpinBox(mask_group)
        min_area.setRange(0, 10_000_000)
        min_area.setValue(int(settings.get("min_area", 0)))
        dilate = QtWidgets.QSpinBox(mask_group)
        dilate.setRange(0, 100)
        dilate.setValue(int(settings.get("dilate", 0)))
        mask_form.addRow("Invert mask", invert)
        mask_form.addRow("Min area", min_area)
        mask_form.addRow("Dilate", dilate)
        root.addWidget(mask_group)
        root.addStretch(1)

        widgets.update(
            {
                "sam_checkpoint": sam_ckpt,
                "sam_model_type": sam_type,
                "delta_type": delta_type,
                "delta_checkpoint": delta_ckpt,
                "middle_dim": middle_dim,
                "scaling_factor": scaling,
                "rank": rank,
                "invert_mask": invert,
                "min_area": min_area,
                "dilate": dilate,
            }
        )

    def _wrap_row(self, layout: QtWidgets.QHBoxLayout) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget(self)
        widget.setLayout(layout)
        return widget


class DinoSettingsForm(QtWidgets.QWidget):
    def __init__(self, settings: dict, widgets: dict[str, QtWidgets.QWidget], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        group = QtWidgets.QGroupBox("GroundingDINO", self)
        form = QtWidgets.QFormLayout(group)
        dino_ckpt = QtWidgets.QLineEdit(str(settings.get("dino_checkpoint") or ""), group)
        dino_btn = QtWidgets.QPushButton("Browse...", group)
        dino_btn.clicked.connect(lambda: browse_file(self, dino_ckpt, "Checkpoint (*.pth);;All files (*)"))
        dino_ckpt_row = QtWidgets.QHBoxLayout()
        dino_ckpt_row.addWidget(dino_ckpt, 1)
        dino_ckpt_row.addWidget(dino_btn)
        dino_cfg = QtWidgets.QComboBox(group)
        dino_cfg.addItem("auto", "auto")
        dino_cfg.addItem("grounding-dino-base", "IDEA-Research/grounding-dino-base")
        dino_cfg.addItem("grounding-dino-tiny", "IDEA-Research/grounding-dino-tiny")
        set_combo_by_data(dino_cfg, str(settings.get("dino_config_id") or "auto"))
        device = QtWidgets.QComboBox(group)
        for value in ("auto", "cuda", "mps", "cpu"):
            device.addItem(value)
        device.setCurrentText(str(settings.get("device") or "auto"))
        text_queries = QtWidgets.QLineEdit(str(settings.get("text_queries") or ""), group)
        box_threshold = QtWidgets.QDoubleSpinBox(group)
        box_threshold.setRange(0.0, 1.0)
        box_threshold.setDecimals(4)
        box_threshold.setSingleStep(0.01)
        box_threshold.setValue(float(settings.get("box_threshold", 0.25)))
        text_threshold = QtWidgets.QDoubleSpinBox(group)
        text_threshold.setRange(0.0, 1.0)
        text_threshold.setDecimals(4)
        text_threshold.setSingleStep(0.01)
        text_threshold.setValue(float(settings.get("text_threshold", 0.25)))
        max_dets = QtWidgets.QSpinBox(group)
        max_dets.setRange(1, 10000)
        max_dets.setValue(int(settings.get("max_dets", 20)))
        form.addRow("Checkpoint", self._wrap_row(dino_ckpt_row))
        form.addRow("Config", dino_cfg)
        form.addRow("Device", device)
        form.addRow("Text queries", text_queries)
        form.addRow("Box threshold", box_threshold)
        form.addRow("Text threshold", text_threshold)
        form.addRow("Max detections", max_dets)
        root.addWidget(group)
        root.addStretch(1)

        widgets.update(
            {
                "dino_checkpoint": dino_ckpt,
                "dino_config_id": dino_cfg,
                "device": device,
                "text_queries": text_queries,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "max_dets": max_dets,
            }
        )

    def _wrap_row(self, layout: QtWidgets.QHBoxLayout) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget(self)
        widget.setLayout(layout)
        return widget


class UnetSettingsForm(QtWidgets.QWidget):
    def __init__(self, settings: dict, widgets: dict[str, QtWidgets.QWidget], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        group = QtWidgets.QGroupBox("UNet", self)
        form = QtWidgets.QFormLayout(group)
        model = QtWidgets.QLineEdit(str(settings.get("unet_model") or ""), group)
        model_btn = QtWidgets.QPushButton("Browse...", group)
        model_btn.clicked.connect(lambda: browse_file(self, model, "Checkpoint (*.pth);;All files (*)"))
        model_row = QtWidgets.QHBoxLayout()
        model_row.addWidget(model, 1)
        model_row.addWidget(model_btn)
        threshold = QtWidgets.QDoubleSpinBox(group)
        threshold.setRange(0.0, 1.0)
        threshold.setDecimals(4)
        threshold.setSingleStep(0.01)
        threshold.setValue(float(settings.get("unet_threshold", 0.5)))
        post = QtWidgets.QCheckBox(group)
        post.setChecked(bool(settings.get("unet_post", True)))
        mode = QtWidgets.QComboBox(group)
        mode.addItems(["tile", "full"])
        mode.setCurrentText(str(settings.get("unet_mode") or "tile"))
        input_size = QtWidgets.QSpinBox(group)
        input_size.setRange(32, 4096)
        input_size.setValue(int(settings.get("unet_input_size", 512)))
        overlap = QtWidgets.QSpinBox(group)
        overlap.setRange(0, 4096)
        overlap.setValue(int(settings.get("unet_overlap", 0)))
        tile_batch = QtWidgets.QSpinBox(group)
        tile_batch.setRange(1, 512)
        tile_batch.setValue(int(settings.get("unet_tile_batch", 4)))
        form.addRow("Model", self._wrap_row(model_row))
        form.addRow("Threshold", threshold)
        form.addRow("Postprocess", post)
        form.addRow("Mode", mode)
        form.addRow("Input size", input_size)
        form.addRow("Overlap", overlap)
        form.addRow("Tile batch", tile_batch)
        root.addWidget(group)
        root.addStretch(1)

        widgets.update(
            {
                "unet_model": model,
                "unet_threshold": threshold,
                "unet_post": post,
                "unet_mode": mode,
                "unet_input_size": input_size,
                "unet_overlap": overlap,
                "unet_tile_batch": tile_batch,
            }
        )

    def _wrap_row(self, layout: QtWidgets.QHBoxLayout) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget(self)
        widget.setLayout(layout)
        return widget
