from __future__ import annotations

from PySide6 import QtGui, QtWidgets


class LogPanel(QtWidgets.QPlainTextEdit):
    """Bottom dock: streaming log output with monospace font."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setPlaceholderText("Log output will appear here…")

        font = QtGui.QFont()
        font.setFamilies(["Menlo", "Consolas", "Cascadia Code", "Monaco", "monospace"])
        font.setPointSize(12)
        self.setFont(font)

        self.setStyleSheet("""
            QPlainTextEdit {
                background: #1a1a1a;
                color: #c8c8c8;
                selection-background-color: rgba(55,140,255,0.3);
                padding: 4px 8px;
            }
        """)
