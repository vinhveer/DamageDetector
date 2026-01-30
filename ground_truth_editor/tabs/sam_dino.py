import os
from pathlib import Path
from dataclasses import asdict
from PySide6 import QtCore, QtWidgets

from constants import (
    DEFAULT_SAM_CHECKPOINT,
    DEFAULT_SAM_OUT_DIR,
    DEFAULT_GDINO_CHECKPOINT,
    DEFAULT_GDINO_CONFIG,
    DEFAULT_TEXT_QUERIES,
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_TEXT_THRESHOLD,
    DEFAULT_MAX_DETS,
    DEFAULT_MIN_AREA,
    DEFAULT_DILATE,
)
from sam_dino.runner import SamDinoParams


class SamDinoTab(QtWidgets.QWidget):
    runRequested = QtCore.Signal(SamDinoParams)
    isolateRequested = QtCore.Signal(SamDinoParams, list, int, bool)  # params, target_labels, outside_val, crop

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid)

        self._sd_sam_ckpt = QtWidgets.QLineEdit(self)
        self._sd_sam_ckpt.setText(DEFAULT_SAM_CHECKPOINT)
        ckpt_browse = QtWidgets.QPushButton("Browse...", self)
        ckpt_browse.clicked.connect(lambda: self._browse_file(self._sd_sam_ckpt, "SAM checkpoint (*.pth);;All files (*)"))

        self._sd_sam_type = QtWidgets.QComboBox(self)
        self._sd_sam_type.addItems(["auto", "vit_b", "vit_l", "vit_h"])

        self._sd_use_delta = QtWidgets.QCheckBox("Use fine-tuned (delta) model", self)
        self._sd_use_delta.setChecked(False)

        self._sd_delta_group = QtWidgets.QGroupBox("Fine-tune (delta)", self)
        self._sd_delta_group.setVisible(False)
        dg = QtWidgets.QGridLayout(self._sd_delta_group)

        self._sd_delta_type = QtWidgets.QComboBox(self._sd_delta_group)
        self._sd_delta_type.addItems(["adapter", "lora", "both"])
        self._sd_delta_type.setCurrentText("lora")

        self._sd_delta_ckpt = QtWidgets.QLineEdit(self._sd_delta_group)
        self._sd_delta_ckpt.setPlaceholderText("auto or path/to/delta.pth")
        self._sd_delta_ckpt.setText("auto")
        delta_browse = QtWidgets.QPushButton("Browse...", self._sd_delta_group)
        delta_browse.clicked.connect(
            lambda: self._browse_file(self._sd_delta_ckpt, "Delta checkpoint (*.pth);;All files (*)")
        )

        self._sd_middle_dim = QtWidgets.QSpinBox(self._sd_delta_group)
        self._sd_middle_dim.setRange(1, 4096)
        self._sd_middle_dim.setValue(32)

        self._sd_scaling_factor = QtWidgets.QDoubleSpinBox(self._sd_delta_group)
        self._sd_scaling_factor.setRange(0.0, 10.0)
        self._sd_scaling_factor.setDecimals(4)
        self._sd_scaling_factor.setSingleStep(0.05)
        self._sd_scaling_factor.setValue(0.2)

        self._sd_rank = QtWidgets.QSpinBox(self._sd_delta_group)
        self._sd_rank.setRange(1, 1024)
        self._sd_rank.setValue(4)

        dr = 0
        dg.addWidget(QtWidgets.QLabel("Delta type"), dr, 0)
        dg.addWidget(self._sd_delta_type, dr, 1, 1, 2)
        dr += 1
        dg.addWidget(QtWidgets.QLabel("Delta checkpoint"), dr, 0)
        dg.addWidget(self._sd_delta_ckpt, dr, 1)
        dg.addWidget(delta_browse, dr, 2)
        dr += 1
        dg.addWidget(QtWidgets.QLabel("Adapter middle_dim"), dr, 0)
        dg.addWidget(self._sd_middle_dim, dr, 1)
        dr += 1
        dg.addWidget(QtWidgets.QLabel("Adapter scaling_factor"), dr, 0)
        dg.addWidget(self._sd_scaling_factor, dr, 1)
        dr += 1
        dg.addWidget(QtWidgets.QLabel("LoRA rank"), dr, 0)
        dg.addWidget(self._sd_rank, dr, 1)
        dg.setColumnStretch(1, 1)

        self._sd_gdino_ckpt = QtWidgets.QLineEdit(self)
        self._sd_gdino_ckpt.setPlaceholderText("path/to/groundingdino.pth or HF model id")
        self._sd_gdino_ckpt.setText(DEFAULT_GDINO_CHECKPOINT)
        gdino_browse = QtWidgets.QPushButton("Browse...", self)
        gdino_browse.clicked.connect(
            lambda: self._browse_file(self._sd_gdino_ckpt, "GroundingDINO checkpoint (*.pth);;All files (*)")
        )

        self._sd_gdino_cfg = QtWidgets.QComboBox(self)
        self._sd_gdino_cfg.addItem("Auto (infer from filename)", "auto")
        self._sd_gdino_cfg.addItem("grounding-dino-base", "IDEA-Research/grounding-dino-base")
        self._sd_gdino_cfg.addItem("grounding-dino-tiny", "IDEA-Research/grounding-dino-tiny")

        self._sd_queries = QtWidgets.QLineEdit(self)
        self._sd_queries.setText(DEFAULT_TEXT_QUERIES)

        self._sd_isolate_labels = QtWidgets.QLineEdit(self)
        self._sd_isolate_labels.setPlaceholderText("Target labels (comma, optional)")

        self._sd_isolate_crop = QtWidgets.QCheckBox("Crop to bbox", self)
        self._sd_isolate_outside_white = QtWidgets.QCheckBox("Outside white (255)", self)

        self._sd_box_thr = QtWidgets.QDoubleSpinBox(self)
        self._sd_box_thr.setRange(0.0, 1.0)
        self._sd_box_thr.setSingleStep(0.01)
        self._sd_box_thr.setValue(DEFAULT_BOX_THRESHOLD)

        self._sd_text_thr = QtWidgets.QDoubleSpinBox(self)
        self._sd_text_thr.setRange(0.0, 1.0)
        self._sd_text_thr.setSingleStep(0.01)
        self._sd_text_thr.setValue(DEFAULT_TEXT_THRESHOLD)

        self._sd_max_dets = QtWidgets.QSpinBox(self)
        self._sd_max_dets.setRange(0, 999)
        self._sd_max_dets.setValue(DEFAULT_MAX_DETS)

        self._sd_invert = QtWidgets.QCheckBox("Invert mask", self)
        self._sd_invert.setChecked(False)

        self._sd_min_area = QtWidgets.QSpinBox(self)
        self._sd_min_area.setRange(0, 10_000_000)
        self._sd_min_area.setValue(DEFAULT_MIN_AREA)

        self._sd_dilate = QtWidgets.QSpinBox(self)
        self._sd_dilate.setRange(0, 50)
        self._sd_dilate.setValue(DEFAULT_DILATE)

        self._sd_outdir = QtWidgets.QLineEdit(self)
        self._sd_outdir.setText(DEFAULT_SAM_OUT_DIR)
        out_browse = QtWidgets.QPushButton("Browse...", self)
        out_browse.clicked.connect(lambda: self._browse_dir(self._sd_outdir))

        r = 0
        grid.addWidget(QtWidgets.QLabel("SAM checkpoint"), r, 0)
        grid.addWidget(self._sd_sam_ckpt, r, 1)
        grid.addWidget(ckpt_browse, r, 2)
        r += 1
        grid.addWidget(QtWidgets.QLabel("SAM type"), r, 0)
        grid.addWidget(self._sd_sam_type, r, 1)
        r += 1
        grid.addWidget(self._sd_use_delta, r, 1, 1, 2)
        r += 1
        grid.addWidget(self._sd_delta_group, r, 0, 1, 3)
        r += 1
        grid.addWidget(QtWidgets.QLabel("GroundingDINO checkpoint / model id"), r, 0)
        grid.addWidget(self._sd_gdino_ckpt, r, 1)
        grid.addWidget(gdino_browse, r, 2)
        r += 1
        grid.addWidget(QtWidgets.QLabel("GroundingDINO config"), r, 0)
        grid.addWidget(self._sd_gdino_cfg, r, 1, 1, 2)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Text queries (comma)"), r, 0)
        grid.addWidget(self._sd_queries, r, 1, 1, 2)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Isolate labels (comma)"), r, 0)
        grid.addWidget(self._sd_isolate_labels, r, 1, 1, 2)
        r += 1
        grid.addWidget(self._sd_isolate_crop, r, 1)
        r += 1
        grid.addWidget(self._sd_isolate_outside_white, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Box thr"), r, 0)
        grid.addWidget(self._sd_box_thr, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Text thr"), r, 0)
        grid.addWidget(self._sd_text_thr, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Max dets"), r, 0)
        grid.addWidget(self._sd_max_dets, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Min component area"), r, 0)
        grid.addWidget(self._sd_min_area, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Dilate iters"), r, 0)
        grid.addWidget(self._sd_dilate, r, 1)
        r += 1
        grid.addWidget(self._sd_invert, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Output dir"), r, 0)
        grid.addWidget(self._sd_outdir, r, 1)
        grid.addWidget(out_browse, r, 2)
        grid.setColumnStretch(1, 1)

        self._sd_use_delta.toggled.connect(self._sd_delta_group.setVisible)
        self._sd_delta_type.currentTextChanged.connect(self._sync_delta_controls)
        self._sd_delta_ckpt.textChanged.connect(self._auto_enable_delta_if_path)
        self._sync_delta_controls(self._sd_delta_type.currentText())

        btn_row = QtWidgets.QHBoxLayout()
        self._sd_run_btn = QtWidgets.QPushButton("Run SAM + DINO", self)
        self._sd_run_btn.clicked.connect(self._on_run_clicked)
        self._sd_isolate_btn = QtWidgets.QPushButton("Tách vật thể", self)
        self._sd_isolate_btn.clicked.connect(self._on_isolate_clicked)
        btn_row.addWidget(self._sd_run_btn)
        btn_row.addWidget(self._sd_isolate_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        layout.addStretch(1)

    def _sync_delta_controls(self, delta_type: str) -> None:
        dt = str(delta_type).strip().lower()
        use_adapter = dt in {"adapter", "both"}
        use_lora = dt in {"lora", "both"}
        self._sd_middle_dim.setEnabled(use_adapter)
        self._sd_scaling_factor.setEnabled(use_adapter)
        self._sd_rank.setEnabled(use_lora)

    def _auto_enable_delta_if_path(self, text: str) -> None:
        t = str(text or "").strip()
        if not t:
            return
        if t.lower() == "auto":
            return
        low = t.lower().replace("\\", "/")
        base = low.rsplit("/", 1)[-1]
        has_adapter = "adapter" in base
        has_lora = "lora" in base
        if has_adapter and has_lora:
            self._sd_delta_type.setCurrentText("both")
        elif has_adapter:
            self._sd_delta_type.setCurrentText("adapter")
        elif has_lora:
            self._sd_delta_type.setCurrentText("lora")
        if not self._sd_use_delta.isChecked():
            self._sd_use_delta.setChecked(True)

    def _browse_file(self, line_edit: QtWidgets.QLineEdit, filter_str: str) -> None:
        start = line_edit.text().strip() or str(Path.cwd())
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", start, filter_str)
        if f:
            line_edit.setText(f)

    def _browse_dir(self, line_edit: QtWidgets.QLineEdit) -> None:
        start = line_edit.text().strip() or str(Path.cwd())
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder", start)
        if d:
            line_edit.setText(d)

    def _build_params(self, queries_override: list[str] = None) -> SamDinoParams:
        ckpt = self._sd_sam_ckpt.text().strip()
        gdino_ckpt = self._sd_gdino_ckpt.text().strip()
        if not gdino_ckpt:
            raise ValueError("GroundingDINO checkpoint is required.")
        if not os.path.exists(gdino_ckpt):
            lower = gdino_ckpt.lower()
            if lower.endswith((".pth", ".pt", ".safetensors", ".bin")):
                raise ValueError(f"GroundingDINO checkpoint not found: {gdino_ckpt}")

        out_dir = self._sd_outdir.text().strip() or DEFAULT_SAM_OUT_DIR
        
        if queries_override is None:
            queries = [q.strip() for q in self._sd_queries.text().split(",") if q.strip()]
            if not queries:
                queries = ["crack"]
        else:
            queries = queries_override

        delta_type = "none"
        delta_ckpt = "auto"
        middle_dim = 32
        scaling_factor = 0.2
        rank = 4

        if self._sd_use_delta.isChecked():
            delta_type = self._sd_delta_type.currentText()
            delta_ckpt = self._sd_delta_ckpt.text().strip() or "auto"
            middle_dim = self._sd_middle_dim.value()
            scaling_factor = self._sd_scaling_factor.value()
            rank = self._sd_rank.value()

        return SamDinoParams(
            sam_checkpoint=ckpt,
            sam_model_type=self._sd_sam_type.currentText(),
            delta_type=delta_type,
            delta_checkpoint=delta_ckpt,
            middle_dim=middle_dim,
            scaling_factor=scaling_factor,
            rank=rank,
            gdino_checkpoint=gdino_ckpt,
            gdino_config_id=self._sd_gdino_cfg.currentData() or "auto",
            text_queries=queries,
            box_threshold=self._sd_box_thr.value(),
            text_threshold=self._sd_text_thr.value(),
            max_dets=self._sd_max_dets.value(),
            overlay_alpha=0.45,
            invert_mask=self._sd_invert.isChecked(),
            sam_min_component_area=self._sd_min_area.value(),
            sam_dilate_iters=self._sd_dilate.value(),
            output_dir=out_dir,
        )

    def _on_run_clicked(self) -> None:
        try:
            params = self._build_params()
            self.runRequested.emit(params)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def _on_isolate_clicked(self) -> None:
        try:
            target_labels = [x.strip() for x in self._sd_isolate_labels.text().split(",") if x.strip()]
            if not target_labels:
                QtWidgets.QMessageBox.warning(self, "Warning", "Please enter >= 1 isolate labels.")
                return

            queries_override = target_labels  # isolate mode uses these as text queries
            params = self._build_params(queries_override=queries_override)
            
            outside_val = 255 if self._sd_isolate_outside_white.isChecked() else 0
            crop = self._sd_isolate_crop.isChecked()
            
            self.isolateRequested.emit(params, target_labels, outside_val, crop)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
