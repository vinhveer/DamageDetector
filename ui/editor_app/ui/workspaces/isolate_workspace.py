from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets


class IsolateWorkspace(QtWidgets.QWidget):
    openIsolateRequested = QtCore.Signal(str)
    openRunRequested = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("Isolated Outputs", self))
        header.addStretch(1)
        self._refresh_btn = QtWidgets.QPushButton("Refresh", self)
        header.addWidget(self._refresh_btn)
        root.addLayout(header)

        self._list = QtWidgets.QTreeWidget(self)
        self._list.setHeaderLabels(["Run", "Workflow", "Image", "Isolate"])
        self._list.itemSelectionChanged.connect(self._update_details)
        self._list.itemDoubleClicked.connect(self._emit_open_isolate)
        root.addWidget(self._list, 1)

        actions = QtWidgets.QHBoxLayout()
        self._open_image_btn = QtWidgets.QPushButton("Open Isolate", self)
        self._open_run_btn = QtWidgets.QPushButton("Open Run Folder", self)
        self._open_image_btn.clicked.connect(self._emit_open_current_isolate)
        self._open_run_btn.clicked.connect(self._emit_open_current_run)
        actions.addWidget(self._open_image_btn)
        actions.addWidget(self._open_run_btn)
        actions.addStretch(1)
        root.addLayout(actions)

        self._details = QtWidgets.QPlainTextEdit(self)
        self._details.setReadOnly(True)
        root.addWidget(self._details, 1)

    def refresh_button(self) -> QtWidgets.QPushButton:
        return self._refresh_btn

    def set_items(self, items: list[dict]) -> None:
        self._list.clear()
        for item_data in items:
            image_name = Path(str(item_data.get("image_path") or "")).name
            isolate_name = Path(str(item_data.get("isolate_path") or "")).name
            item = QtWidgets.QTreeWidgetItem(
                [
                    str(item_data.get("run_id") or ""),
                    str(item_data.get("workflow") or ""),
                    image_name,
                    isolate_name,
                ]
            )
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, dict(item_data))
            self._list.addTopLevelItem(item)
        if self._list.topLevelItemCount() > 0:
            self._list.setCurrentItem(self._list.topLevelItem(0))

    def _selected_data(self) -> dict | None:
        item = self._list.currentItem()
        if item is None:
            return None
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        return dict(data) if isinstance(data, dict) else None

    def _update_details(self) -> None:
        data = self._selected_data()
        if not data:
            self._details.clear()
            return
        lines = [
            f"run_id: {data.get('run_id')}",
            f"workflow: {data.get('workflow')}",
            f"status: {data.get('status')}",
            f"created_at: {data.get('created_at')}",
            f"prompt: {data.get('prompt')}",
            f"isolate_action: {data.get('isolate_action')}",
            f"image_path: {data.get('image_path')}",
            f"isolate_path: {data.get('isolate_path')}",
            f"mask_path: {data.get('mask_path')}",
            f"run_dir: {data.get('run_dir')}",
        ]
        self._details.setPlainText("\n".join(lines))

    def _emit_open_current_isolate(self) -> None:
        data = self._selected_data()
        if data and data.get("isolate_path"):
            self.openIsolateRequested.emit(str(data.get("isolate_path")))

    def _emit_open_isolate(self, item: QtWidgets.QTreeWidgetItem) -> None:
        if item is None:
            return
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(data, dict) and data.get("isolate_path"):
            self.openIsolateRequested.emit(str(data.get("isolate_path")))

    def _emit_open_current_run(self) -> None:
        data = self._selected_data()
        if data and data.get("run_dir"):
            self.openRunRequested.emit(str(data.get("run_dir")))
