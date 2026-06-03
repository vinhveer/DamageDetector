from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from PySide6 import QtCore

from ..paths import repo_root, semi_labeling_dir


STEP_MODULES = {
    "step04": "steps.step04_core.main",
    "step05": "steps.step05_proto.main",
    "step06": "steps.step06_reliability.main",
    "step07": "steps.step07_decision.main",
    "step08": "steps.step08_classifier.main",
    "step09": "steps.step09_self_train.main",
    "export_dataset": "tools.export_dataset",
}


def resolve_python() -> str:
    if os.name == "nt":
        candidates = [
            os.environ.get("SEMI_LABELING_PYTHON"),
            repo_root() / ".venv" / "Scripts" / "python.exe",
            semi_labeling_dir() / ".venv" / "Scripts" / "python.exe",
        ]
    else:
        candidates = [
            os.environ.get("SEMI_LABELING_PYTHON"),
            repo_root() / ".venv" / "bin" / "python",
            semi_labeling_dir() / ".venv" / "bin" / "python",
        ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return str(path)
    return "python"


def flags_to_argv(flags: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in (flags or {}).items():
        if not re.match(r"^--[a-z0-9-]+$", str(key)):
            raise ValueError(f"Unsafe flag name: {key}")
        if value is False or value is None:
            continue
        if value is True:
            argv.append(str(key))
            continue
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"Unsafe flag value for {key}")
        argv.extend([str(key), str(value)])
    return argv


class StepRunner(QtCore.QObject):
    output = QtCore.Signal(str)
    started = QtCore.Signal(str)
    finished = QtCore.Signal(int)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._process: QtCore.QProcess | None = None
        self._finished = False

    def run(self, step: str, flags: dict[str, Any] | None = None) -> None:
        module = STEP_MODULES.get(step)
        if not module:
            self.output.emit(f"Unknown step: {step}\n")
            self.finished.emit(-1)
            return
        try:
            argv = ["-m", module, *flags_to_argv(flags or {})]
        except Exception as exc:
            self.output.emit(f"{exc}\n")
            self.finished.emit(-1)
            return

        proc = QtCore.QProcess(self)
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONPATH", str(semi_labeling_dir()))
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        proc.setProcessEnvironment(env)
        proc.setWorkingDirectory(str(semi_labeling_dir()))
        proc.readyReadStandardOutput.connect(lambda: self._drain(proc.readAllStandardOutput()))
        proc.readyReadStandardError.connect(lambda: self._drain(proc.readAllStandardError()))
        proc.errorOccurred.connect(self._on_error)
        proc.finished.connect(lambda code, _status: self._finish(code))
        self._process = proc
        self._finished = False
        python = resolve_python()
        self.output.emit("$ " + " ".join([python, *argv]) + "\n")
        self.started.emit(step)
        proc.start(python, argv)

    def _on_error(self, error: QtCore.QProcess.ProcessError) -> None:
        if error == QtCore.QProcess.ProcessError.FailedToStart:
            self.output.emit("\n[spawn error] failed to start process\n")
            self._finish(-1)

    def _drain(self, data: QtCore.QByteArray) -> None:
        self.output.emit(bytes(data).decode("utf-8", errors="replace"))

    def _finish(self, code: int) -> None:
        if self._finished:
            return
        self._finished = True
        self.output.emit(f"\n[exit {code}]\n")
        self._process = None
        self.finished.emit(int(code))

    def stop(self) -> None:
        if self._process and self._process.state() != QtCore.QProcess.ProcessState.NotRunning:
            self._process.kill()
