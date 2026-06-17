from __future__ import annotations

from abc import ABC, abstractmethod

from PySide6 import QtCore


class Command(ABC):
    label: str = ""

    @abstractmethod
    def redo(self) -> None: ...

    @abstractmethod
    def undo(self) -> None: ...


class UndoStack(QtCore.QObject):
    pushed = QtCore.Signal(object)
    cursorChanged = QtCore.Signal(int)
    cleanChanged = QtCore.Signal(bool)

    def __init__(self, parent: QtCore.QObject | None = None, limit: int = 200) -> None:
        super().__init__(parent)
        self._undo: list[Command] = []
        self._redo: list[Command] = []
        self._limit = int(limit)

    def push(self, command: Command) -> None:
        command.redo()
        self._undo.append(command)
        if len(self._undo) > self._limit:
            self._undo.pop(0)
        self._redo.clear()
        self.pushed.emit(command)
        self.cursorChanged.emit(len(self._undo))

    def undo(self) -> bool:
        if not self._undo:
            return False
        command = self._undo.pop()
        command.undo()
        self._redo.append(command)
        self.cursorChanged.emit(len(self._undo))
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        command = self._redo.pop()
        command.redo()
        self._undo.append(command)
        self.cursorChanged.emit(len(self._undo))
        return True

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
        self.cursorChanged.emit(0)

    def history(self) -> list[Command]:
        return list(self._undo)
