from __future__ import annotations

from pathlib import Path

from PySide6 import QtWidgets


def set_combo_by_data(combo: QtWidgets.QComboBox, data_value: str) -> None:
    for i in range(combo.count()):
        if str(combo.itemData(i)) == str(data_value):
            combo.setCurrentIndex(i)
            return


def browse_file(parent: QtWidgets.QWidget, line_edit: QtWidgets.QLineEdit, filter_str: str) -> None:
    start = line_edit.text().strip() or str(Path.cwd())
    f, _ = QtWidgets.QFileDialog.getOpenFileName(parent, "Select file", start, filter_str)
    if f:
        line_edit.setText(f)


def browse_dir(parent: QtWidgets.QWidget, line_edit: QtWidgets.QLineEdit) -> None:
    start = line_edit.text().strip() or str(Path.cwd())
    d = QtWidgets.QFileDialog.getExistingDirectory(parent, "Select folder", start)
    if d:
        line_edit.setText(d)


def append_log(widget: QtWidgets.QPlainTextEdit, text: str) -> None:
    widget.appendPlainText(str(text))
