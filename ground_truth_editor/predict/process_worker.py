from __future__ import annotations

import json
import sys
import threading
import traceback as tb
from typing import Any, Callable


class WorkerProtocol:
    def __init__(self) -> None:
        self._print_lock = threading.Lock()
        self._stop_events: dict[int, threading.Event] = {}
        self._running: set[int] = set()

    def emit(self, obj: dict[str, Any]) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        with self._print_lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def log_fn(self, call_id: int) -> Callable[[str], None]:
        def _log(msg: str) -> None:
            self.emit({"type": "log", "id": int(call_id), "text": str(msg)})

        return _log

    def stop_checker(self, call_id: int) -> Callable[[], bool]:
        ev = self._stop_events.setdefault(int(call_id), threading.Event())
        return ev.is_set

    def request_stop(self, call_id: int) -> None:
        ev = self._stop_events.setdefault(int(call_id), threading.Event())
        ev.set()

    def _job_done(self, call_id: int) -> None:
        self._running.discard(int(call_id))
        self._stop_events.pop(int(call_id), None)

    def spawn_job(self, call_id: int, fn: Callable[[], Any]) -> None:
        if self._running:
            self.emit(
                {
                    "type": "error",
                    "id": int(call_id),
                    "error": {"type": "Busy", "message": "Service is busy (one job at a time)."},
                }
            )
            return

        self._running.add(int(call_id))

        def _run() -> None:
            try:
                result = fn()
                if isinstance(result, dict) and result.get("stopped"):
                    self.emit({"type": "stopped", "id": int(call_id)})
                elif self.stop_checker(int(call_id))():
                    self.emit({"type": "stopped", "id": int(call_id)})
                else:
                    self.emit({"type": "result", "id": int(call_id), "result": result})
            except Exception as e:
                if e.__class__.__name__ == "StopRequested" or str(e) == "Stopped":
                    self.emit({"type": "stopped", "id": int(call_id)})
                else:
                    self.emit(
                        {
                            "type": "error",
                            "id": int(call_id),
                            "error": {"type": e.__class__.__name__, "message": str(e), "traceback": tb.format_exc()},
                        }
                    )
            finally:
                self._job_done(int(call_id))

        threading.Thread(target=_run, name=f"job:{call_id}", daemon=True).start()


def run_worker(dispatch: Callable[[WorkerProtocol, int, str, dict[str, Any]], None]) -> int:
    proto = WorkerProtocol()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            proto.emit({"type": "log", "id": None, "text": line})
            continue
        if not isinstance(obj, dict):
            continue

        mtype = str(obj.get("type") or "")
        if mtype == "stop":
            try:
                proto.request_stop(int(obj.get("id")))
            except Exception:
                pass
            continue

        if mtype != "call":
            continue

        call_id = int(obj.get("id"))
        method = str(obj.get("method") or "")
        params = obj.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        try:
            dispatch(proto, call_id, method, params)
        except Exception as e:
            proto.emit({"type": "error", "id": call_id, "error": {"type": e.__class__.__name__, "message": str(e), "traceback": tb.format_exc()}})
    return 0
