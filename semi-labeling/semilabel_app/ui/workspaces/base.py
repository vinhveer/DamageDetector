from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class PlaceholderWorkspace(QtWidgets.QWidget):
    def __init__(self, title: str, message: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(10)
        heading = QtWidgets.QLabel(title, self)
        font = heading.font()
        font.setPointSize(18)
        font.setBold(True)
        heading.setFont(font)
        text = QtWidgets.QLabel(message, self)
        text.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(text)
        layout.addStretch(1)


class FormRow(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(str)

    def __init__(self, label: str, value: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.label = QtWidgets.QLabel(label, self)
        self.label.setFixedWidth(92)
        self.edit = QtWidgets.QLineEdit(value, self)
        self.edit.textChanged.connect(self.valueChanged)
        layout.addWidget(self.label)
        layout.addWidget(self.edit, 1)

    def text(self) -> str:
        return self.edit.text()

    def set_text(self, value: str) -> None:
        self.edit.setText(str(value))
