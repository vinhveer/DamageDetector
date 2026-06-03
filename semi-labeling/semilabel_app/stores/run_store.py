from __future__ import annotations

from PySide6 import QtCore


class RunStore(QtCore.QObject):
    logChanged = QtCore.Signal()
    statusChanged = QtCore.Signal(str)
    stepChanged = QtCore.Signal(str, int)
    runningChanged = QtCore.Signal(bool)
    errorRaised = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.log_text = ""
        self.running = False
        self.current_step = ""
        self.last_status = "Ready"
        self.step_codes: dict[str, int] = {}

    def clear_log(self) -> None:
        self.log_text = ""
        self.logChanged.emit()

    def append_log(self, text: str) -> None:
        self.log_text += str(text)
        self.logChanged.emit()

    def set_running(self, value: bool) -> None:
        if self.running == bool(value):
            return
        self.running = bool(value)
        self.runningChanged.emit(self.running)

    def set_status(self, text: str) -> None:
        self.last_status = str(text)
        self.statusChanged.emit(self.last_status)

    def set_step(self, step: str, code: int = -999) -> None:
        self.current_step = str(step)
        if code != -999:
            self.step_codes[self.current_step] = int(code)
        self.stepChanged.emit(self.current_step, int(code))
