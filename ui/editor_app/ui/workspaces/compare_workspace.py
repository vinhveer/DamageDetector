from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ui.editor_app.ui.components.compare_results_panel import CompareResultsPanel


class CompareWorkspace(QtWidgets.QWidget):
    selectedRunChanged = QtCore.Signal(str)
    groundTruthDirChanged = QtCore.Signal(str)
    affixChanged = QtCore.Signal(str)
    compareRequested = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        controls = QtWidgets.QHBoxLayout()
        self._run_combo = QtWidgets.QComboBox(self)
        self._gt_edit = QtWidgets.QLineEdit(self)
        self._gt_edit.setPlaceholderText("Ground-truth folder")
        self._suffix_edit = QtWidgets.QLineEdit(self)
        self._suffix_edit.setPlaceholderText("affix, e.g. mask")
        self._browse_btn = QtWidgets.QPushButton("Browse GT", self)
        self._compare_btn = QtWidgets.QPushButton("Compare", self)
        self._browse_btn.clicked.connect(self._browse_gt)
        self._compare_btn.clicked.connect(self._emit_compare)
        self._run_combo.currentIndexChanged.connect(self._emit_selected_run_changed)
        self._gt_edit.textChanged.connect(self.groundTruthDirChanged.emit)
        self._suffix_edit.textChanged.connect(self.affixChanged.emit)
        controls.addWidget(QtWidgets.QLabel("Run", self))
        controls.addWidget(self._run_combo, 1)
        controls.addWidget(self._gt_edit, 2)
        controls.addWidget(self._suffix_edit, 1)
        controls.addWidget(self._browse_btn)
        controls.addWidget(self._compare_btn)
        root.addLayout(controls)

        self._panel = CompareResultsPanel(self)
        root.addWidget(self._panel, 1)

    def set_runs(self, runs: list[tuple[str, str]]) -> None:
        current = self._run_combo.currentData()
        self._run_combo.blockSignals(True)
        self._run_combo.clear()
        for label, run_dir in runs:
            self._run_combo.addItem(label, run_dir)
        if current:
            index = self._run_combo.findData(current)
            if index >= 0:
                self._run_combo.setCurrentIndex(index)
        self._run_combo.blockSignals(False)

    def set_compare_config(self, *, gt_dir: str, affix: str, selected_run_dir: str | None = None) -> None:
        self._run_combo.blockSignals(True)
        self._gt_edit.blockSignals(True)
        self._suffix_edit.blockSignals(True)
        self._gt_edit.setText(str(gt_dir or ""))
        self._suffix_edit.setText(str(affix or ""))
        self._gt_edit.blockSignals(False)
        self._suffix_edit.blockSignals(False)
        if selected_run_dir:
            index = self._run_combo.findData(selected_run_dir)
            if index >= 0:
                self._run_combo.setCurrentIndex(index)
        self._run_combo.blockSignals(False)

    def set_results(self, results: list[dict]) -> None:
        self._panel.set_results(results)

    def _browse_gt(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select GT Folder")
        if path:
            self._gt_edit.setText(path)

    def _emit_selected_run_changed(self) -> None:
        self.selectedRunChanged.emit(str(self._run_combo.currentData() or ""))

    def _emit_compare(self) -> None:
        self.compareRequested.emit(str(self._run_combo.currentData() or ""))
