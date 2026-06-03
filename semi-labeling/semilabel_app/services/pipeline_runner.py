from __future__ import annotations

from PySide6 import QtCore

from ..domain.models import ChainStep
from .step_runner import StepRunner


class PipelineRunner(QtCore.QObject):
    step_started = QtCore.Signal(int, str)
    output = QtCore.Signal(int, str)
    step_finished = QtCore.Signal(int, int)
    chain_finished = QtCore.Signal(bool)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._steps: list[ChainStep] = []
        self._index = -1
        self._runner: StepRunner | None = None

    def run(self, steps: list[ChainStep]) -> None:
        if not steps:
            self.chain_finished.emit(True)
            return
        self._steps = list(steps)
        self._index = -1
        self._start_next()

    def _start_next(self) -> None:
        self._index += 1
        if self._index >= len(self._steps):
            self.chain_finished.emit(True)
            return
        step = self._steps[self._index]
        runner = StepRunner(self)
        runner.output.connect(lambda text, idx=self._index: self.output.emit(idx, text))
        runner.finished.connect(self._on_finished)
        self._runner = runner
        self.step_started.emit(self._index, step.key)
        runner.run(step.key, step.flags)

    def _on_finished(self, code: int) -> None:
        self.step_finished.emit(self._index, int(code))
        if code == 0:
            self._start_next()
        else:
            self.chain_finished.emit(False)

    def stop(self) -> None:
        if self._runner:
            self._runner.stop()
