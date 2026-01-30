from pathlib import Path
from PySide6 import QtCore, QtWidgets

from constants import (
    DEFAULT_UNET_MODEL_PATH,
    DEFAULT_UNET_OUT_DIR,
    DEFAULT_UNET_THRESHOLD,
    DEFAULT_UNET_INPUT_SIZE,
    DEFAULT_UNET_TILE_BATCH,
    DEFAULT_UNET_OVERLAP,
)
from predict_unet import UnetParams


class UnetTab(QtWidgets.QWidget):
    runRequested = QtCore.Signal(UnetParams)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid)

        self._unet_model_edit = QtWidgets.QLineEdit(self)
        self._unet_model_edit.setText(DEFAULT_UNET_MODEL_PATH)
        model_browse = QtWidgets.QPushButton("Browse...", self)
        model_browse.clicked.connect(lambda: self._browse_file(self._unet_model_edit, "Model (*.pth);;All files (*)"))

        self._unet_outdir_edit = QtWidgets.QLineEdit(self)
        self._unet_outdir_edit.setText(DEFAULT_UNET_OUT_DIR)
        out_browse = QtWidgets.QPushButton("Browse...", self)
        out_browse.clicked.connect(lambda: self._browse_dir(self._unet_outdir_edit))

        self._unet_threshold = QtWidgets.QDoubleSpinBox(self)
        self._unet_threshold.setRange(0.0, 1.0)
        self._unet_threshold.setSingleStep(0.01)
        self._unet_threshold.setValue(DEFAULT_UNET_THRESHOLD)

        self._unet_post = QtWidgets.QCheckBox("Apply postprocessing", self)
        self._unet_post.setChecked(True)

        self._unet_mode = QtWidgets.QComboBox(self)
        self._unet_mode.addItems(["tile", "letterbox", "resize"])

        self._unet_input_size = QtWidgets.QSpinBox(self)
        self._unet_input_size.setRange(64, 4096)
        self._unet_input_size.setValue(DEFAULT_UNET_INPUT_SIZE)

        self._unet_overlap = QtWidgets.QSpinBox(self)
        self._unet_overlap.setRange(0, 4096)
        self._unet_overlap.setValue(DEFAULT_UNET_OVERLAP)
        self._unet_overlap.setToolTip("0 = recommended (input_size//2)")

        self._unet_tile_batch = QtWidgets.QSpinBox(self)
        self._unet_tile_batch.setRange(1, 128)
        self._unet_tile_batch.setValue(DEFAULT_UNET_TILE_BATCH)

        r = 0
        grid.addWidget(QtWidgets.QLabel("Model (.pth)"), r, 0)
        grid.addWidget(self._unet_model_edit, r, 1)
        grid.addWidget(model_browse, r, 2)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Output dir"), r, 0)
        grid.addWidget(self._unet_outdir_edit, r, 1)
        grid.addWidget(out_browse, r, 2)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Threshold"), r, 0)
        grid.addWidget(self._unet_threshold, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Mode"), r, 0)
        grid.addWidget(self._unet_mode, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Input size"), r, 0)
        grid.addWidget(self._unet_input_size, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Tile overlap"), r, 0)
        grid.addWidget(self._unet_overlap, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Tile batch size"), r, 0)
        grid.addWidget(self._unet_tile_batch, r, 1)
        r += 1
        grid.addWidget(self._unet_post, r, 1)
        grid.setColumnStretch(1, 1)

        btn_row = QtWidgets.QHBoxLayout()
        self._unet_run_btn = QtWidgets.QPushButton("Run UNet", self)
        self._unet_run_btn.clicked.connect(self._on_run_clicked)
        btn_row.addWidget(self._unet_run_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        layout.addStretch(1)

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

    def _on_run_clicked(self) -> None:
        params = UnetParams(
            model_path=self._unet_model_edit.text().strip(),
            output_dir=self._unet_outdir_edit.text().strip(),
            threshold=self._unet_threshold.value(),
            use_post_processing=self._unet_post.isChecked(),
            mode=self._unet_mode.currentText(),
            input_size=self._unet_input_size.value(),
            tile_overlap=self._unet_overlap.value(),
            batch_size=self._unet_tile_batch.value(),
        )
        self.runRequested.emit(params)
