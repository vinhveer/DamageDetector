from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from uuid import uuid4

from PySide6 import QtCore


ROOT = Path(__file__).resolve().parents[2]
TMP_ROOT = ROOT / ".tmp"


class CliJob(QtCore.QObject):
    """Run one workflow as a standalone subprocess.

    Each job owns a working directory at `.tmp/<workflow>/<job_id>/` and is
    killed as a process group on cancel/close so child model workers die too.
    """

    log = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)
    finished = QtCore.Signal(int)
    failed = QtCore.Signal(str)

    def __init__(self, *, workflow: str, module: str, args: list[str] | None = None, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.workflow = str(workflow).strip() or "workflow"
        self.job_id = uuid4().hex
        self.job_dir = TMP_ROOT / self.workflow / self.job_id
        self.module = str(module)
        self.args = list(args or [])
        self._proc: QtCore.QProcess | None = None
        self._buf = ""
        self._cancelled = False

    def start(self) -> None:
        self.job_dir.mkdir(parents=True, exist_ok=True)
        proc = QtCore.QProcess(self)
        proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        proc.setWorkingDirectory(str(ROOT))
        proc.readyReadStandardOutput.connect(self._on_output)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)

        final_args = ["-m", self.module, *self.args]
        self._proc = proc
        self.log.emit(f"Job {self.job_id}: {self.workflow} tmp={self.job_dir}")
        self.log.emit(f"Launching: {sys.executable} {' '.join(final_args)}")
        proc.start(sys.executable, final_args)

    def cancel(self) -> None:
        self._cancelled = True
        proc = self._proc
        if proc is None or proc.state() == QtCore.QProcess.ProcessState.NotRunning:
            return
        pid = int(proc.processId())
        self.log.emit(f"Cancel: killing process group of pid {pid}...")
        killed = False
        if hasattr(os, "killpg"):
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                killed = True
            except (ProcessLookupError, PermissionError):
                killed = False
        if not killed:
            proc.kill()

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.state() != QtCore.QProcess.ProcessState.NotRunning

    @QtCore.Slot()
    def _on_output(self) -> None:
        if self._proc is None:
            return
        data = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                continue
            if line.startswith("PROGRESS "):
                try:
                    done, total = line[len("PROGRESS "):].split("/", 1)
                    self.progress.emit(int(done), int(total))
                except ValueError:
                    pass
                continue
            self.log.emit(line)

    @QtCore.Slot()
    def _on_error(self, error: QtCore.QProcess.ProcessError) -> None:
        if self._cancelled:
            return
        self.failed.emit(f"Process error: {error}")

    @QtCore.Slot()
    def _on_finished(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus) -> None:
        del exit_status
        if self._cancelled:
            self.failed.emit("Cancelled by user.")
            return
        self.finished.emit(int(exit_code))
