from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets


class ExplorerPanel(QtWidgets.QWidget):
    addFolderImagesRequested = QtCore.Signal()
    imageClicked = QtCore.Signal(str)
    imageActivated = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._images: list[str] = []
        self._path_to_row: dict[str, int] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        top = QtWidgets.QHBoxLayout()
        self._filter = QtWidgets.QLineEdit(self)
        self._filter.setPlaceholderText("Type to filter")
        self._filter.textChanged.connect(self._apply_filter)
        top.addWidget(self._filter, 1)
        self._select_all_btn = QtWidgets.QToolButton(self)
        self._select_all_btn.setText("Select all")
        self._select_all_btn.clicked.connect(self.select_all_visible)
        top.addWidget(self._select_all_btn)
        layout.addLayout(top)

        self._list = QtWidgets.QListWidget(self)
        self._list.setUniformItemSizes(True)
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self._list.itemClicked.connect(self._on_clicked)
        self._list.itemActivated.connect(self._on_activated)
        self._list.itemDoubleClicked.connect(self._on_activated)
        self._list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self._list, 1)

    def set_images(self, images: list[str]) -> None:
        self._images = list(images)
        self._path_to_row = {}
        self._list.clear()
        for idx, path in enumerate(self._images):
            item = QtWidgets.QListWidgetItem(os.path.basename(path))
            item.setToolTip(path)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
            self._list.addItem(item)
            self._path_to_row[path] = idx
        self._apply_filter(self._filter.text())

    def select_path(self, path: str) -> None:
        row = self._path_to_row.get(path)
        if row is None:
            return
        item = self._list.item(row)
        if item is None or item.isHidden():
            return
        self._list.blockSignals(True)
        self._list.setCurrentRow(row)
        self._list.scrollToItem(item, QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter)
        self._list.blockSignals(False)

    def focus_filter(self) -> None:
        self._filter.setFocus()
        self._filter.selectAll()

    def select_all_visible(self) -> None:
        self._list.blockSignals(True)
        self._list.clearSelection()
        for index in range(self._list.count()):
            item = self._list.item(index)
            if item is not None and not item.isHidden():
                item.setSelected(True)
        self._list.blockSignals(False)

    def _apply_filter(self, text: str) -> None:
        query = str(text or "").strip().lower()
        for index in range(self._list.count()):
            item = self._list.item(index)
            if item is None:
                continue
            if not query:
                item.setHidden(False)
                continue
            path = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "").lower()
            item.setHidden(query not in item.text().lower() and query not in path)

    def _on_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        path = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if path:
            self.imageClicked.emit(str(path))

    def _on_activated(self, item: QtWidgets.QListWidgetItem) -> None:
        path = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if path:
            self.imageActivated.emit(str(path))

    def _show_context_menu(self, point: QtCore.QPoint) -> None:
        item = self._list.itemAt(point)

        menu = QtWidgets.QMenu(self)
        act_add_folder = menu.addAction("Add Folder Images...")
        menu.addSeparator()
        if item is None:
            chosen = menu.exec(self._list.mapToGlobal(point))
            if chosen == act_add_folder:
                self.addFolderImagesRequested.emit()
            return

        path = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        if not path or not os.path.exists(path):
            chosen = menu.exec(self._list.mapToGlobal(point))
            if chosen == act_add_folder:
                self.addFolderImagesRequested.emit()
            return

        act_copy = menu.addAction("Copy Path")
        act_reveal = menu.addAction("Reveal")
        chosen = menu.exec(self._list.mapToGlobal(point))
        if chosen == act_add_folder:
            self.addFolderImagesRequested.emit()
        elif chosen == act_copy:
            QtWidgets.QApplication.clipboard().setText(path)
        elif chosen == act_reveal:
            self._reveal(path)

    def _reveal(self, path: str) -> None:
        file_path = Path(path)
        folder = str(file_path.parent)
        if os.name == "nt":
            subprocess.run(["explorer", "/select,", os.path.normpath(str(file_path))], check=False)
            return
        if sys.platform == "darwin":
            subprocess.run(["open", "-R", str(file_path)], check=False)
            return
        if shutil.which("xdg-open"):
            subprocess.run(["xdg-open", folder], check=False)
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(folder))
