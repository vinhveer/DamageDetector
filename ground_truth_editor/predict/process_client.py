from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from predict.ipc import parse_json_line


class ServiceCrashed(RuntimeError):
    pass


@dataclass(frozen=True)
class CallResult:
    result: Any


@dataclass(frozen=True)
class CallError:
    message: str
    error_type: str | None = None
    traceback: str | None = None


class JsonServiceProcess:
    def __init__(self, *, module: str, cwd: str | None = None, env: dict[str, str] | None = None) -> None:
        self._module = module
        self._cwd = cwd
        self._env = env
        self._proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._q: queue.Queue[dict[str, Any]] = queue.Queue()
        self._lock = threading.Lock()
        self._next_id = 1
        self._stopping: set[int] = set()

    def _default_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        repo_root = Path(__file__).resolve().parents[2]
        app_root = repo_root / "ground_truth_editor"
        py_path = env.get("PYTHONPATH", "")
        parts = [str(app_root), str(repo_root)]
        if py_path:
            parts.append(py_path)
        env["PYTHONPATH"] = os.pathsep.join(parts)
        return env

    def ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return

        self.close()
        env = dict(self._default_env())
        if self._env:
            env.update(self._env)

        cwd = self._cwd or str(Path(__file__).resolve().parents[2])
        self._proc = subprocess.Popen(
            [sys.executable, "-m", self._module],
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        self._reader = threading.Thread(target=self._read_loop, name=f"svc-reader:{self._module}", daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        try:
            for line in self._proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = parse_json_line(line)
                except Exception:
                    obj = {"type": "log", "id": None, "text": line}
                self._q.put(obj)
        finally:
            self._q.put({"type": "_eof"})

    def _send(self, obj: dict[str, Any]) -> None:
        self.ensure_started()
        assert self._proc is not None
        assert self._proc.stdin is not None
        data = json.dumps(obj, ensure_ascii=False) + "\n"
        with self._lock:
            try:
                self._proc.stdin.write(data)
                self._proc.stdin.flush()
            except Exception as e:
                raise ServiceCrashed(str(e)) from e

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        log_fn: Callable[[str], None] | None = None,
        stop_checker: Callable[[], bool] | None = None,
        poll_s: float = 0.15,
        timeout_s: float | None = None,
    ) -> Any:
        self.ensure_started()
        call_id = self._next_id
        self._next_id += 1
        self._stopping.discard(call_id)
        self._send({"type": "call", "id": call_id, "method": method, "params": params or {}})

        # Defensive default: warmup may hang on Windows (AV/file-lock/tokenizer parsing).
        # If it does, we prefer timing out and restarting the worker instead of freezing the UI forever.
        if timeout_s is None and str(method or "").strip().lower() == "warmup":
            timeout_s = 180.0

        started = time.time()
        while True:
            if timeout_s is not None and (time.time() - started) > float(timeout_s):
                if log_fn is not None:
                    try:
                        log_fn(f"ERROR: '{method}' timed out after {int(timeout_s)}s. Restarting service process...")
                    except Exception:
                        pass
                self.close()
                raise TimeoutError(f"Service call '{method}' timed out after {int(timeout_s)}s")

            if stop_checker is not None and stop_checker() and call_id not in self._stopping:
                self._stopping.add(call_id)
                try:
                    self._send({"type": "stop", "id": call_id})
                except Exception:
                    pass

            try:
                msg = self._q.get(timeout=poll_s)
            except queue.Empty:
                continue

            mtype = str(msg.get("type") or "")
            if mtype == "_eof":
                raise ServiceCrashed(f"Service {self._module} ended unexpectedly.")

            mid = msg.get("id")
            if mtype == "log":
                if log_fn is not None and (mid is None or int(mid) == int(call_id)):
                    log_fn(str(msg.get("text") or ""))
                continue

            if mid is None or int(mid) != int(call_id):
                continue

            if mtype == "result":
                return msg.get("result")
            if mtype == "stopped":
                return {"stopped": True}
            if mtype == "error":
                err = msg.get("error") or {}
                raise RuntimeError(f"{err.get('type')}: {err.get('message')}")

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2)
                except Exception:
                    self._proc.kill()
        except Exception:
            pass
        self._proc = None
