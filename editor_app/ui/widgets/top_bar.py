from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class TopBar(QtWidgets.QWidget):
    workspaceRequested = QtCore.Signal(str)
    predictRequested = QtCore.Signal()
    predictRoiRequested = QtCore.Signal()
    isolateRequested = QtCore.Signal()
    saveImageRequested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)

        self._workspace_order: list[str] = []
        self._workspace_tabs = QtWidgets.QTabBar(self)
        self._workspace_tabs.setDrawBase(False)
        self._workspace_tabs.setExpanding(False)
        self._workspace_tabs.setUsesScrollButtons(True)
        self._workspace_tabs.currentChanged.connect(self._emit_workspace_changed)
        for key, label in (
            ("editor", "Editor"),
            ("runs", "Runs"),
            ("history", "History"),
            ("compare", "Compare"),
            ("isolate", "Isolate"),
            ("settings", "Settings"),
        ):
            self._workspace_tabs.addTab(label)
            self._workspace_order.append(key)
        row.addWidget(self._workspace_tabs)

        row.addStretch(1)

        self._predict_btn = QtWidgets.QPushButton("Run Predict", self)
        self._predict_roi_btn = QtWidgets.QPushButton("Run ROI Predict", self)
        self._isolate_btn = QtWidgets.QPushButton("Run Isolate", self)
        self._save_image_btn = QtWidgets.QPushButton("Save Image As", self)
        self._predict_btn.clicked.connect(self.predictRequested.emit)
        self._predict_roi_btn.clicked.connect(self.predictRoiRequested.emit)
        self._isolate_btn.clicked.connect(self.isolateRequested.emit)
        self._save_image_btn.clicked.connect(self.saveImageRequested.emit)
        row.addWidget(self._predict_btn)
        row.addWidget(self._predict_roi_btn)
        row.addWidget(self._isolate_btn)
        row.addWidget(self._save_image_btn)
        root.addLayout(row)

    def _emit_workspace_changed(self, index: int) -> None:
        if 0 <= index < len(self._workspace_order):
            self.workspaceRequested.emit(self._workspace_order[index])

    def set_active_workspace(self, name: str) -> None:
        current = str(name or "")
        try:
            index = self._workspace_order.index(current)
        except ValueError:
            index = 0
        self._workspace_tabs.blockSignals(True)
        self._workspace_tabs.setCurrentIndex(index)
        self._workspace_tabs.blockSignals(False)

    def set_current_path(self, path: str | None) -> None:
        _ = path
