from __future__ import annotations

import os
from dataclasses import dataclass

from PySide6 import QtCore, QtWidgets


@dataclass(frozen=True)
class ExplorerSelection:
    clicked_path: str | None = None
    activated_path: str | None = None


class ExplorerPanel(QtWidgets.QWidget):
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
        self._filter.setPlaceholderText("Type to filter (Ctrl+K)")
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

    def has_images(self) -> bool:
        return bool(self._images)

    def images(self) -> list[str]:
        return list(self._images)

    def visible_count(self) -> int:
        visible = 0
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it is not None and not it.isHidden():
                visible += 1
        return visible

    def focus_filter(self) -> None:
        self._filter.setFocus()
        self._filter.selectAll()

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
        if self._list.currentRow() == row:
            return
        self._list.blockSignals(True)
        self._list.setCurrentRow(row)
        self._list.scrollToItem(item, QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter)
        self._list.blockSignals(False)

    def navigate(self, delta: int, current_path: str | None) -> str | None:
        if not self._images or self._list.count() == 0:
            return None
        start_row = self._list.currentRow()
        if start_row < 0 and current_path:
            start_row = self._path_to_row.get(current_path, -1)
        if start_row < 0:
            start_row = 0

        step = 1 if delta >= 0 else -1
        row = start_row + step
        while 0 <= row < self._list.count():
            item = self._list.item(row)
            if item is not None and not item.isHidden():
                found = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if found:
                    self._list.setCurrentRow(row)
                    return str(found)
            row += step
        return None

    def select_all_visible(self) -> None:
        if self._list.count() == 0:
            return
        self._list.blockSignals(True)
        self._list.clearSelection()
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item is not None and not item.isHidden():
                item.setSelected(True)
        self._list.blockSignals(False)

    def _apply_filter(self, text: str) -> None:
        query = str(text or "").strip().lower()
        total = self._list.count()
        for i in range(total):
            item = self._list.item(i)
            if item is None:
                continue
            if not query:
                item.setHidden(False)
                continue
            name = item.text().lower()
            path = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "").lower()
            match = query in name or query in path
            item.setHidden(not match)

    def _on_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        found = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if found:
            self.imageClicked.emit(str(found))

    def _on_activated(self, item: QtWidgets.QListWidgetItem) -> None:
        found = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if found:
            self.imageActivated.emit(str(found))

    def _show_context_menu(self, point: QtCore.QPoint) -> None:
        """Show context menu for list items using customContextMenuRequested."""
        item = self._list.itemAt(point)
        if not item:
            return

        path = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        if not path or not os.path.exists(path):
            return

        menu = QtWidgets.QMenu(self)
        
        act_rename = menu.addAction("Rename")
        act_delete = menu.addAction("Delete")
        menu.addSeparator()
        act_copy_path = menu.addAction("Copy Path")
        act_reveal = menu.addAction("Reveal in Explorer")

        # Map point to global for exec
        global_point = self._list.mapToGlobal(point)
        val = menu.exec(global_point)
        
        if val == act_rename:
            self._rename_item(item, path)
        elif val == act_delete:
            self._delete_item(item, path)
        elif val == act_copy_path:
            QtWidgets.QApplication.clipboard().setText(path)
        elif val == act_reveal:
            self._reveal_in_explorer(path)

    def _rename_item(self, item: QtWidgets.QListWidgetItem, path: str) -> None:
        base = os.path.basename(path)
        new_name, ok = QtWidgets.QInputDialog.getText(self, "Rename", "New name:", text=base)
        if not ok or not new_name.strip() or new_name == base:
            return
            
        new_name = new_name.strip()
        dn = os.path.dirname(path)
        new_path = os.path.join(dn, new_name)
        
        try:
            os.rename(path, new_path)
            # Update item and internal list
            item.setText(new_name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, new_path)
            item.setToolTip(new_path)
            
            # Update self._images and map
            try:
                idx = self._images.index(path)
                self._images[idx] = new_path
            except ValueError:
                pass
            
            self._path_to_row.pop(path, None)
            self._path_to_row[new_path] = self._list.row(item)
            
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Rename failed: {e}")

    def _delete_item(self, item: QtWidgets.QListWidgetItem, path: str) -> None:
        res = QtWidgets.QMessageBox.question(
            self, 
            "Delete", 
            f"Are you sure you want to delete '{os.path.basename(path)}'?\nThis cannot be undone.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        if res != QtWidgets.QMessageBox.StandardButton.Yes:
            return
            
        try:
            os.remove(path)
            row = self._list.row(item)
            self._list.takeItem(row) # This removes it from widget
            
            try:
                self._images.remove(path)
            except ValueError:
                pass
            self._path_to_row.pop(path, None)
            
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Delete failed: {e}")

    def _reveal_in_explorer(self, path: str) -> None:
        import subprocess
        # Windows specific
        if os.name == 'nt':
            subprocess.run(['explorer', '/select,', os.path.normpath(path)])

