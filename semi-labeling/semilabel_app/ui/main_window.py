"""Main window shell: connection bar, tabbed pages, global shortcuts.

The heavy lifting now lives in ``ui/pages`` and ``ui/widgets``.  This module
only wires those pieces together and owns the shared ``ImageService`` plus the
connection settings.  ``ConnectDialog`` is re-exported here for backwards
compatibility with ``app.py`` and any external imports.
"""
from __future__ import annotations

from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from ..config.defaults import DEFAULT_SETTINGS
from ..services.image_service import ImageService
from ..services.settings_service import SettingsService
from .connect_dialog import ConnectDialog
from .options_dialog import OptionsDialog
from .pages import ImageOverviewPage, PrototypePage, ReviewPage

__all__ = ["ConnectDialog", "MainWindow"]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings_service: SettingsService, settings: dict[str, Any]) -> None:
        super().__init__()
        self._settings_service = settings_service
        self._settings = settings
        self._db_path = str(settings.get("db_path") or "")
        self._image_root = str(settings.get("image_root") or "")
        self._run_id = str(settings.get("run_id") or "myrun")
        self._reviewer = str(settings.get("reviewer") or "")
        self._notes = str(settings.get("notes") or "")
        self._model_name = str(settings.get("model_name") or "facebook/dinov2-giant")
        self.image_service = ImageService(max_items=1000, max_full_items=8, max_thumb_items=900, parent=self)
        self.setWindowTitle("Semi-labeling Review")
        self.resize(1420, 860)

        self.tabs = QtWidgets.QTabWidget(self)
        self.tabs.setDocumentMode(True)
        self.before_page = ImageOverviewPage(self)
        self.review_page = ReviewPage(self, cleaned=False)
        self.prototype_page = PrototypePage(self)
        self.tabs.addTab(self.before_page, "Before")
        self.tabs.addTab(self.review_page, "Review")
        self.tabs.addTab(self.prototype_page, "Prototype")
        self.tabs.currentChanged.connect(self.ensure_current_tab_loaded)
        self.setCentralWidget(self.tabs)

        self._build_menu()
        self._build_status_bar()
        self._install_global_shortcuts()
        QtCore.QTimer.singleShot(0, self.ensure_current_tab_loaded)

    @property
    def _pages(self) -> tuple[QtWidgets.QWidget, ...]:
        return (self.before_page, self.review_page, self.prototype_page)

    # -- chrome ------------------------------------------------------------
    def _build_menu(self) -> None:
        menu = self.menuBar()
        data_menu = menu.addMenu("&Data")
        connect_action = data_menu.addAction("Change connection…")
        connect_action.setShortcut(QtGui.QKeySequence("Ctrl+K"))
        connect_action.triggered.connect(self.change_connection)
        options_action = data_menu.addAction("Tab options…")
        options_action.setShortcut(QtGui.QKeySequence("Ctrl+,"))
        options_action.triggered.connect(self.open_current_options)

    def _build_status_bar(self) -> None:
        self._connection_label = QtWidgets.QLabel(self)
        self._connection_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.statusBar().addPermanentWidget(self._connection_label)
        self._refresh_connection_label()
        self.statusBar().showMessage("Ready — app only reads DB and writes JSON")

    def set_busy(self, busy: bool) -> None:
        cursor = QtCore.Qt.CursorShape.BusyCursor if busy else QtCore.Qt.CursorShape.ArrowCursor
        self.setCursor(cursor)

    def open_current_options(self) -> None:
        page = self.tabs.currentWidget()
        spec_fn = getattr(page, "options_spec", None)
        apply_fn = getattr(page, "apply_options", None)
        if not callable(spec_fn) or not callable(apply_fn):
            self.status("This tab has no options.")
            return
        spec = spec_fn()
        if not spec:
            self.status("This tab has no options.")
            return
        title = f"{getattr(page, 'title_text', 'Tab')} options"
        values = OptionsDialog.edit(title, spec, self)
        if values is not None:
            apply_fn(values)

    def _install_global_shortcuts(self) -> None:
        for tab_index, key in enumerate(("Ctrl+1", "Ctrl+2", "Ctrl+3")):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(lambda index=tab_index: self.tabs.setCurrentIndex(index))

    def ensure_current_tab_loaded(self, *_args: object) -> None:
        page = self.tabs.currentWidget()
        if hasattr(page, "ensure_loaded"):
            page.ensure_loaded()

    def _refresh_connection_label(self) -> None:
        if hasattr(self, "_connection_label"):
            self._connection_label.setText(
                f"DB: {self._db_path}   |   Images: {self._image_root}   |   Run: {self._run_id}"
            )

    # -- connection accessors ---------------------------------------------
    def db_path(self) -> str:
        return self._db_path

    def image_root(self) -> str:
        return self._image_root

    def run_id(self) -> str:
        return self._run_id or "myrun"

    def reviewer(self) -> str:
        return self._reviewer

    def notes(self) -> str:
        return self._notes

    def model_name(self) -> str:
        return self._model_name or "facebook/dinov2-giant"

    @QtCore.Slot()
    def change_connection(self) -> None:
        dialog = ConnectDialog(self._settings, self)
        dialog.db.setText(self._db_path)
        dialog.images.setText(self._image_root)
        dialog.run.setText(self._run_id)
        dialog.reviewer.setText(self._reviewer)
        dialog.notes.setText(self._notes)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.apply_connection(dialog.payload())

    def apply_connection(self, payload: dict[str, Any]) -> None:
        self._db_path = str(payload.get("db_path") or "")
        self._image_root = str(payload.get("image_root") or "")
        self._run_id = str(payload.get("run_id") or "myrun")
        self._reviewer = str(payload.get("reviewer") or "")
        self._notes = str(payload.get("notes") or "")
        self._settings.update(payload)
        self._refresh_connection_label()
        self.save_settings()
        for page in self._pages:
            if hasattr(page, "reset"):
                page.reset()
            for attr in ("items", "visible_items", "groups", "members"):
                if hasattr(page, attr):
                    setattr(page, attr, [])
            if hasattr(page, "pending"):
                page.pending.clear()
            if hasattr(page, "decisions"):
                page.decisions.clear()
            if hasattr(page, "_order_cache"):
                page._order_cache.clear()
            view = getattr(page, getattr(page, "list_attr", "list"), None)
            if view is not None:
                view.set_payloads([], lambda _payload: "", None)
            if hasattr(page, "image"):
                page.image.clear()
        self.ensure_current_tab_loaded()

    def save_settings(self) -> None:
        settings = dict(DEFAULT_SETTINGS)
        settings.update({
            "db_path": self.db_path(),
            "image_root": self.image_root(),
            "run_id": self.run_id(),
            "reviewer": self.reviewer(),
            "notes": self.notes(),
            "model_name": self.model_name(),
        })
        self._settings_service.save({"settings": settings})
        self.status("Settings saved")

    def status(self, text: str) -> None:
        self.statusBar().showMessage(text, 5000)

    def error(self, text: str) -> None:
        QtWidgets.QMessageBox.warning(self, "Semi-labeling", text)
        self.status(text)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.save_settings()
        super().closeEvent(event)
