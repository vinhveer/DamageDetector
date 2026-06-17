from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ui.core.commands import Command, UndoStack


class HistoryPanel(QtWidgets.QWidget):
    """Bottom dock tab: undo history list."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.list = QtWidgets.QListWidget(self)
        self.list.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.list.setAlternatingRowColors(True)
        self.list.setSpacing(0)
        self.list.setUniformItemSizes(True)
        font = QtGui.QFont("Menlo, Consolas, monospace")
        font.setPointSize(12)
        self.list.setFont(font)
        root.addWidget(self.list, 1)

        self._stack: UndoStack | None = None

    def attach(self, stack: UndoStack) -> None:
        self._stack = stack
        stack.pushed.connect(self._refresh)
        stack.cursorChanged.connect(self._refresh)
        self._refresh()

    def _refresh(self, *_args: object) -> None:
        self.list.clear()
        if self._stack is None:
            return
        history = self._stack.history()
        for i, cmd in enumerate(history):
            label = cmd.label or cmd.__class__.__name__
            item = QtWidgets.QListWidgetItem(f"  {i + 1:>3}.  {label}")
            self.list.addItem(item)
        if history:
            self.list.scrollToBottom()
