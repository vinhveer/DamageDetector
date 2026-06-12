"""Shared base class for review/prototype/images/groups pages.

Centralises the boilerplate that used to be copy-pasted across every page:

* a reference to the owning window and the shared ``ImageService``
* a per-page ``DbExecutor`` so list/detail queries run off the GUI thread
* lazy thumbnail loading driven by the list's visible rows
* full-image loading with a "current key" guard so out-of-order async
  decodes never overwrite the image the user is now looking at
* a small busy spinner shown while a background query is in flight
"""
from __future__ import annotations

from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from ...services.async_db import DbExecutor


class BasePage(QtWidgets.QWidget):
    #: Subclasses set this so MainWindow can reset state on reconnect.
    list_attr = "list"

    def __init__(self, window: "Any") -> None:  # MainWindow, avoid import cycle
        super().__init__(window)
        self.window = window
        self._image_service = window.image_service
        self.db = DbExecutor(self)
        self._current_image_key: tuple[str, str] | None = None
        self._loaded_once = False
        self.db.busyChanged.connect(self._on_busy_changed)

    # -- lifecycle ---------------------------------------------------------
    def ensure_loaded(self) -> None:
        if not self._loaded_once:
            self.load()

    def reset(self) -> None:
        """Forget cached state so the next ``ensure_loaded`` reloads."""
        self._loaded_once = False
        self._current_image_key = None

    def load(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    # -- busy indicator ----------------------------------------------------
    def _on_busy_changed(self, _channel: str, busy: bool) -> None:
        # Route busy state to the window (status bar / cursor) so individual
        # pages do not need their own spinner widget.
        setter = getattr(self.window, "set_busy", None)
        if callable(setter):
            setter(bool(busy))

    # -- image helpers -----------------------------------------------------
    def _show_full_image(self, image_service_key: tuple[str, str], image_path: str, image_widget) -> bool:
        """Display ``image_path`` in ``image_widget``; load async if uncached.

        Returns ``False`` if the path is empty (caller should clear the view).
        """
        if not image_path:
            self._current_image_key = None
            return False
        self._current_image_key = image_service_key
        cached = self._image_service.cached(image_service_key)
        if cached is not None:
            image_widget.set_pixmap(cached)
        else:
            self._image_service.load_image(image_service_key, image_path, size=0)
        return True

    @QtCore.Slot(object, str)
    def on_image_failed(self, key: object, message: str) -> None:
        if key == self._current_image_key:
            self.window.status(f"Image load failed: {message}")
