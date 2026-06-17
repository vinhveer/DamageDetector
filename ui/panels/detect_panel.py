from __future__ import annotations

from PySide6 import QtCore, QtWidgets


def _make_form() -> QtWidgets.QFormLayout:
    form = QtWidgets.QFormLayout()
    form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
    form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
    form.setVerticalSpacing(6)
    form.setHorizontalSpacing(8)
    form.setContentsMargins(8, 4, 8, 4)
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


class _GdinoOptions(QtWidgets.QWidget):
    """GroundingDINO specific options."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        form = _make_form()

        self.checkpoint = QtWidgets.QLineEdit(self)
        self.checkpoint.setPlaceholderText("HuggingFace id or local path")
        self.checkpoint.setText("IDEA-Research/grounding-dino-base")
        form.addRow("Checkpoint", self.checkpoint)

        self.box_threshold = QtWidgets.QDoubleSpinBox(self)
        self.box_threshold.setRange(0.001, 1.0)
        self.box_threshold.setDecimals(3)
        self.box_threshold.setSingleStep(0.01)
        self.box_threshold.setValue(0.25)
        form.addRow("Box thresh", self.box_threshold)

        self.text_threshold = QtWidgets.QDoubleSpinBox(self)
        self.text_threshold.setRange(0.001, 1.0)
        self.text_threshold.setDecimals(3)
        self.text_threshold.setSingleStep(0.01)
        self.text_threshold.setValue(0.25)
        form.addRow("Text thresh", self.text_threshold)

        self.max_dets = QtWidgets.QSpinBox(self)
        self.max_dets.setRange(1, 500)
        self.max_dets.setValue(20)
        form.addRow("Max dets", self.max_dets)

        self.tile_scales = QtWidgets.QLineEdit(self)
        self.tile_scales.setPlaceholderText("e.g. 1.0,0.5")
        self.tile_scales.setText("1.0")
        form.addRow("Tile scales", self.tile_scales)

        self.setLayout(form)


class _StableDinoOptions(QtWidgets.QWidget):
    """StableDINO specific options."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        form = _make_form()

        self.checkpoint = QtWidgets.QLineEdit(self)
        self.checkpoint.setPlaceholderText("Path to StableDINO .pth")
        form.addRow("Checkpoint", _path_picker(self.checkpoint, self, "StableDINO", "Checkpoint (*.pth)"))

        self.config = QtWidgets.QLineEdit(self)
        self.config.setPlaceholderText("Path to config .py (optional)")
        form.addRow("Config", _path_picker(self.config, self, "Config", "Python (*.py)"))

        self.conf = QtWidgets.QDoubleSpinBox(self)
        self.conf.setRange(0.001, 1.0)
        self.conf.setDecimals(3)
        self.conf.setSingleStep(0.01)
        self.conf.setValue(0.05)
        form.addRow("Confidence", self.conf)

        self.max_dets = QtWidgets.QSpinBox(self)
        self.max_dets.setRange(1, 500)
        self.max_dets.setValue(20)
        form.addRow("Max dets", self.max_dets)

        self.setLayout(form)


class _YoloOptions(QtWidgets.QWidget):
    """YOLO specific options."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        form = _make_form()

        self.checkpoint = QtWidgets.QLineEdit(self)
        self.checkpoint.setPlaceholderText("Optional YOLO best.pt; empty = auto model_with_inference")
        form.addRow("Checkpoint", _path_picker(self.checkpoint, self, "YOLO checkpoint", "Checkpoint (*.pt)"))

        self.conf = QtWidgets.QDoubleSpinBox(self)
        self.conf.setRange(0.001, 1.0)
        self.conf.setDecimals(3)
        self.conf.setSingleStep(0.01)
        self.conf.setValue(0.10)
        form.addRow("Confidence", self.conf)

        self.iou = QtWidgets.QDoubleSpinBox(self)
        self.iou.setRange(0.001, 1.0)
        self.iou.setDecimals(3)
        self.iou.setSingleStep(0.01)
        self.iou.setValue(0.45)
        form.addRow("IoU", self.iou)

        self.imgsz = QtWidgets.QSpinBox(self)
        self.imgsz.setRange(64, 2048)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(768)
        form.addRow("Image size", self.imgsz)

        self.max_dets = QtWidgets.QSpinBox(self)
        self.max_dets.setRange(1, 500)
        self.max_dets.setValue(50)
        form.addRow("Max dets", self.max_dets)

        self.setLayout(form)


class DetectPanel(QtWidgets.QWidget):
    """Standalone detect panel — stacked options per detector backend."""

    runRequested = QtCore.Signal()
    cancelRequested = QtCore.Signal()
    filtersChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Common config: detector + filters
        common = _make_form()

        self.detector_combo = QtWidgets.QComboBox(self)
        self.detector_combo.addItem("GroundingDINO", "gdino")
        self.detector_combo.addItem("StableDINO", "stabledino")
        self.detector_combo.addItem("YOLO", "yolo")
        self.detector_combo.currentIndexChanged.connect(self._on_detector_changed)

        self.min_score = QtWidgets.QDoubleSpinBox(self)
        self.min_score.setRange(0.0, 1.0)
        self.min_score.setDecimals(3)
        self.min_score.setSingleStep(0.01)
        self.min_score.setValue(0.0)
        self.min_score.valueChanged.connect(self.filtersChanged)

        self.class_filter = QtWidgets.QComboBox(self)
        self.class_filter.addItem("All classes", "all")
        for cls in ("crack", "mold", "stain", "spall"):
            self.class_filter.addItem(cls.title(), cls)
        self.class_filter.currentTextChanged.connect(self.filtersChanged)

        common.addRow("Detector", self.detector_combo)
        common.addRow("Show score ≥", self.min_score)
        common.addRow("Class", self.class_filter)
        outer.addLayout(common)

        # Divider
        sep = QtWidgets.QFrame(self)
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet("color: rgba(255,255,255,0.07);")
        outer.addWidget(sep)

        # Backend-specific options
        self.gdino_options = _GdinoOptions(self)
        self.stabledino_options = _StableDinoOptions(self)
        self.yolo_options = _YoloOptions(self)

        self._stack = QtWidgets.QStackedWidget(self)
        self._stack.addWidget(self.gdino_options)        # idx 0 = gdino
        self._stack.addWidget(self.stabledino_options)   # idx 1 = stabledino
        self._stack.addWidget(self.yolo_options)         # idx 2 = yolo
        outer.addWidget(self._stack)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(8, 2, 8, 6)
        btn_row.setSpacing(6)
        self.run_button = QtWidgets.QPushButton("▶  Run Detect", self)
        self.cancel_button = QtWidgets.QPushButton("Cancel", self)
        self.cancel_button.setEnabled(False)
        self.run_button.clicked.connect(self.runRequested)
        self.cancel_button.clicked.connect(self.cancelRequested)
        btn_row.addWidget(self.run_button, 2)
        btn_row.addWidget(self.cancel_button, 1)
        outer.addLayout(btn_row)

        # Results table
        self.table = QtWidgets.QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(["ROI", "Class", "Score", "Box"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        outer.addWidget(self.table, 1)

    def _on_detector_changed(self, idx: int) -> None:
        self._stack.setCurrentIndex(idx)

    # Compat for legacy mixin code
    @property
    def run_conf(self) -> QtWidgets.QDoubleSpinBox:
        """Legacy: returns the confidence spinbox of the current backend."""
        if self.detector_combo.currentData() == "gdino":
            return self.gdino_options.box_threshold
        if self.detector_combo.currentData() == "yolo":
            return self.yolo_options.conf
        return self.stabledino_options.conf
