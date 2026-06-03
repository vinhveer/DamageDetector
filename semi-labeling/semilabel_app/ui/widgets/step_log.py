from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class StepLog(QtWidgets.QWidget):
    stopRequested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.status = QtWidgets.QLabel("Sẵn sàng", self)
        self.status.setObjectName("StepStatus")
        self.log = QtWidgets.QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        mono = QtGui.QFont("Cascadia Mono, Consolas, monospace")
        mono.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        mono.setPointSize(11)
        self.log.setFont(mono)
        self.stop_btn = QtWidgets.QPushButton("Dừng", self)
        self.stop_btn.clicked.connect(self.stopRequested)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.status, 1)
        top.addWidget(self.stop_btn)
        root.addLayout(top)
        root.addWidget(self.log, 1)

    def clear(self) -> None:
        self.log.clear()

    def append(self, text: str) -> None:
        self.log.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        self.log.insertPlainText(str(text))
        self.log.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    def set_status(self, text: str) -> None:
        self.status.setText(str(text))
