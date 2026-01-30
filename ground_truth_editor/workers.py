from __future__ import annotations

import os
import sys
import threading
import json
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

from PySide6 import QtCore

from sam_dino.runner import SamDinoParams, SamDinoRunner
from predict_unet import UnetParams, UnetRunner


class WorkerBase(QtCore.QObject):
    log = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    finished = QtCore.Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._stop_event = threading.Event()

    @QtCore.Slot()
    def stop(self) -> None:
        self._stop_event.set()
        self.log.emit("Stop requested...")

    def _stop_checker(self) -> bool:
        return self._stop_event.is_set()


class UnetWorker(WorkerBase):
    def __init__(self, runner: UnetRunner, image_path: str, params: UnetParams) -> None:
        super().__init__()
        self._runner = runner
        self._image_path = image_path
        self._params = params

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self.log.emit(f"UNet: image={self._image_path}")
            self.log.emit(f"UNet: model={self._params.model_path}")
            details = self._runner.run(
                self._image_path,
                self._params,
                stop_checker=self._stop_checker,
                log_fn=self.log.emit,
            )
            self.finished.emit(details)
        except Exception as e:
            if e.__class__.__name__ == "StopRequested" or str(e) == "Stopped":
                self.log.emit("STOPPED")
                self.finished.emit({"stopped": True})
                return
            self.failed.emit(str(e))


class SamDinoWorker(WorkerBase):
    def __init__(self, runner: SamDinoRunner, image_path: str, params: SamDinoParams) -> None:
        super().__init__()
        self._runner = runner
        self._image_path = image_path
        self._params = params
        self._proc: subprocess.Popen[str] | None = None

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self.log.emit(f"SAM+DINO: image={self._image_path}")
            self.log.emit(f"SAM+DINO: sam_checkpoint={self._params.sam_checkpoint}")
            self.log.emit(f"SAM+DINO: gdino_checkpoint={self._params.gdino_checkpoint}")
            self.log.emit(f"SAM+DINO: gdino_config_id={self._params.gdino_config_id}")

            repo_root = Path(__file__).resolve().parents[1]
            payload = {
                "mode": "run",
                "image_path": self._image_path,
                "params": asdict(self._params),
            }

            with tempfile.TemporaryDirectory(prefix="sam_dino_") as td:
                payload_path = os.path.join(td, "payload.json")
                out_path = os.path.join(td, "result.json")
                with open(payload_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                    f.write("\n")

                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                env.setdefault("TOKENIZERS_PARALLELISM", "false")
                py_path = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{py_path}" if py_path else str(repo_root)

                # Assuming sam_dino.cli is importable from repo_root
                cmd = [sys.executable, "-m", "sam_dino.cli", "--payload", payload_path, "--output", out_path]
                self._proc = subprocess.Popen(
                    cmd,
                    cwd=str(repo_root),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                last_lines: list[str] = []
                assert self._proc.stdout is not None
                for line in self._proc.stdout:
                    if self._stop_checker():
                        break
                    msg = line.rstrip("\n")
                    last_lines.append(msg)
                    if len(last_lines) > 60:
                        last_lines.pop(0)
                    self.log.emit(msg)

                if self._stop_checker():
                    try:
                        if self._proc.poll() is None:
                            self._proc.terminate()
                            self._proc.wait(timeout=5)
                    except Exception:
                        try:
                            if self._proc.poll() is None:
                                self._proc.kill()
                        except Exception:
                            pass
                    self.finished.emit({"stopped": True})
                    return

                rc = self._proc.wait()
                self._proc = None
                if rc != 0 and not os.path.isfile(out_path):
                    tail = "\n".join(last_lines[-10:])
                    raise RuntimeError(f"SAM+DINO subprocess exited with code {rc}\n\nLast output:\n{tail}")
                if not os.path.isfile(out_path):
                    tail = "\n".join(last_lines[-10:])
                    raise RuntimeError(f"SAM+DINO subprocess did not produce result.json\n\nLast output:\n{tail}")

                with open(out_path, "r", encoding="utf-8") as f:
                    details = json.load(f)
                if isinstance(details, dict) and details.get("error"):
                    raise RuntimeError(f"{details.get('error_type')}: {details.get('error')}")
                self.finished.emit(details)
        except Exception as e:
            if e.__class__.__name__ == "StopRequested" or str(e) == "Stopped":
                self.log.emit("STOPPED")
                self.finished.emit({"stopped": True})
                return
            self.failed.emit(str(e))


class SamDinoIsolateWorker(WorkerBase):
    def __init__(
        self,
        runner: SamDinoRunner,
        image_path: str,
        params: SamDinoParams,
        *,
        target_labels: list[str],
        outside_value: int,
        crop_to_bbox: bool,
    ) -> None:
        super().__init__()
        self._runner = runner
        self._image_path = image_path
        self._params = params
        self._target_labels = target_labels
        self._outside_value = outside_value
        self._crop_to_bbox = crop_to_bbox
        self._proc: subprocess.Popen[str] | None = None

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self.log.emit(f"SAM+DINO isolate: image={self._image_path}")
            self.log.emit(f"SAM+DINO isolate: sam_checkpoint={self._params.sam_checkpoint}")
            self.log.emit(f"SAM+DINO isolate: gdino_checkpoint={self._params.gdino_checkpoint}")
            self.log.emit(f"SAM+DINO isolate: gdino_config_id={self._params.gdino_config_id}")
            repo_root = Path(__file__).resolve().parents[1]
            payload = {
                "mode": "isolate",
                "image_path": self._image_path,
                "params": asdict(self._params),
                "target_labels": list(self._target_labels),
                "outside_value": int(self._outside_value),
                "crop_to_bbox": bool(self._crop_to_bbox),
            }

            with tempfile.TemporaryDirectory(prefix="sam_dino_") as td:
                payload_path = os.path.join(td, "payload.json")
                out_path = os.path.join(td, "result.json")
                with open(payload_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                    f.write("\n")

                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                env.setdefault("TOKENIZERS_PARALLELISM", "false")
                py_path = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{py_path}" if py_path else str(repo_root)

                cmd = [sys.executable, "-m", "sam_dino.cli", "--payload", payload_path, "--output", out_path]
                self._proc = subprocess.Popen(
                    cmd,
                    cwd=str(repo_root),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                last_lines: list[str] = []
                assert self._proc.stdout is not None
                for line in self._proc.stdout:
                    if self._stop_checker():
                        break
                    msg = line.rstrip("\n")
                    last_lines.append(msg)
                    if len(last_lines) > 60:
                        last_lines.pop(0)
                    self.log.emit(msg)

                if self._stop_checker():
                    try:
                        if self._proc.poll() is None:
                            self._proc.terminate()
                            self._proc.wait(timeout=5)
                    except Exception:
                        try:
                            if self._proc.poll() is None:
                                self._proc.kill()
                        except Exception:
                            pass
                    self.finished.emit({"stopped": True})
                    return

                rc = self._proc.wait()
                self._proc = None
                if rc != 0 and not os.path.isfile(out_path):
                    tail = "\n".join(last_lines[-10:])
                    raise RuntimeError(f"SAM+DINO subprocess exited with code {rc}\n\nLast output:\n{tail}")
                if not os.path.isfile(out_path):
                    tail = "\n".join(last_lines[-10:])
                    raise RuntimeError(f"SAM+DINO subprocess did not produce result.json\n\nLast output:\n{tail}")

                with open(out_path, "r", encoding="utf-8") as f:
                    details = json.load(f)
                if isinstance(details, dict) and details.get("error"):
                    raise RuntimeError(f"{details.get('error_type')}: {details.get('error')}")
                self.finished.emit(details)
        except Exception as e:
            if e.__class__.__name__ == "StopRequested" or str(e) == "Stopped":
                self.log.emit("STOPPED")
                self.finished.emit({"stopped": True})
                return
            self.failed.emit(str(e))
