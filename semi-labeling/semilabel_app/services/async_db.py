"""Background database executor for the semi-labeling app.

Every ``db_service`` call is potentially slow (SQLite over a network share,
large result sets).  Running them on the GUI thread freezes the UI while the
reviewer is scrolling/deciding.  ``DbExecutor`` runs each call on the shared
``QThreadPool`` and delivers the result back on the GUI thread through signals.

Two delivery guarantees matter for a responsive review workflow:

* **latest-wins** — when the user changes the filter quickly, only the most
  recent request for a given ``channel`` should update the UI; stale results are
  dropped (compared by a monotonically increasing token per channel).
* **safe teardown** — if the window closes while a query is in flight, emitting
  to a deleted receiver must not crash; Qt drops queued signals to dead
  receivers, and the worker guards the emit.
"""
from __future__ import annotations

import itertools
from typing import Any, Callable

from PySide6 import QtCore


class _DbSignals(QtCore.QObject):
    done = QtCore.Signal(str, int, object)   # channel, token, result
    failed = QtCore.Signal(str, int, str)    # channel, token, message


class _DbTask(QtCore.QRunnable):
    def __init__(self, channel: str, token: int, fn: Callable[..., Any], args: tuple, kwargs: dict, signals: _DbSignals) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._channel = channel
        self._token = token
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._signals = signals

    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:  # noqa: BLE001 - report any failure to the UI
            try:
                self._signals.failed.emit(self._channel, self._token, str(exc))
            except RuntimeError:
                pass
            return
        try:
            self._signals.done.emit(self._channel, self._token, result)
        except RuntimeError:
            # Receiver/application torn down mid-flight; nothing to deliver.
            pass


class DbExecutor(QtCore.QObject):
    """Run blocking db_service calls off the GUI thread.

    Usage::

        self.db = DbExecutor(self)
        self.db.subscribe("queue", on_ok, on_err)
        self.db.submit("queue", db_service.list_queue, db, run, root, ...)

    ``on_ok(result)`` and ``on_err(message)`` are invoked on the GUI thread, and
    only for the most recent submission on that channel.
    """

    result = QtCore.Signal(str, object)   # channel, result (latest-wins only)
    error = QtCore.Signal(str, str)       # channel, message (latest-wins only)
    busyChanged = QtCore.Signal(str, bool)  # channel, is_busy

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._signals = _DbSignals()
        self._signals.done.connect(self._on_done)
        self._signals.failed.connect(self._on_failed)
        self._pool = QtCore.QThreadPool.globalInstance()
        self._counter = itertools.count(1)
        self._latest: dict[str, int] = {}
        self._ok_handlers: dict[str, list[Callable[[Any], None]]] = {}
        self._err_handlers: dict[str, list[Callable[[str], None]]] = {}

    def subscribe(
        self,
        channel: str,
        on_ok: Callable[[Any], None] | None = None,
        on_err: Callable[[str], None] | None = None,
    ) -> None:
        if on_ok is not None:
            self._ok_handlers.setdefault(channel, []).append(on_ok)
        if on_err is not None:
            self._err_handlers.setdefault(channel, []).append(on_err)

    def submit(self, channel: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> int:
        """Queue ``fn(*args, **kwargs)`` on the pool for ``channel``.

        Returns the request token.  Any earlier in-flight request on the same
        channel becomes stale and its result is ignored.
        """
        token = next(self._counter)
        self._latest[channel] = token
        self.busyChanged.emit(channel, True)
        self._pool.start(_DbTask(channel, token, fn, args, kwargs, self._signals))
        return token

    def is_stale(self, channel: str, token: int) -> bool:
        return self._latest.get(channel) != token

    @QtCore.Slot(str, int, object)
    def _on_done(self, channel: str, token: int, result: object) -> None:
        if self.is_stale(channel, token):
            return
        self.busyChanged.emit(channel, False)
        for handler in self._ok_handlers.get(channel, []):
            handler(result)
        self.result.emit(channel, result)

    @QtCore.Slot(str, int, str)
    def _on_failed(self, channel: str, token: int, message: str) -> None:
        if self.is_stale(channel, token):
            return
        self.busyChanged.emit(channel, False)
        for handler in self._err_handlers.get(channel, []):
            handler(message)
        self.error.emit(channel, message)
