from __future__ import annotations

import os
import sys

from PySide6 import QtCore, QtGui, QtWidgets

from .config.defaults import DEFAULT_SETTINGS, migrate_settings
from .services.settings_service import SettingsService
from .ui.main_window import ConnectDialog, MainWindow


# Keep the look native (Fusion).  The only custom rules are the coloured
# decision buttons, which carry an accent via the ``accent`` dynamic property;
# everything else inherits the platform palette so the UI stays calm and
# uncluttered.
_ACCENT_STYLE = """
QPushButton[accent="crack"]  { background: #4a90d9; color: white; font-weight: 600; }
QPushButton[accent="mold"]   { background: #27ae60; color: white; font-weight: 600; }
QPushButton[accent="spall"]  { background: #f39c12; color: white; font-weight: 600; }
QPushButton[accent="reject"] { background: #e74c3c; color: white; font-weight: 600; }
QPushButton[accent="primary"]{ background: #2563eb; color: white; font-weight: 600; }
QPushButton:checked          { border: 2px solid #1f2937; }
"""


def _apply_palette(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    app.setStyleSheet(_ACCENT_STYLE)


def _load_settings(settings_service: SettingsService) -> dict:
    persisted = settings_service.load()
    settings = dict(DEFAULT_SETTINGS)
    settings.update(migrate_settings(dict(persisted.get("settings") or {})))
    return settings


def build_window(*, interactive_connect: bool = True) -> MainWindow | None:
    settings_service = SettingsService()
    settings = _load_settings(settings_service)
    if interactive_connect:
        dialog = ConnectDialog(settings)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        settings.update(dialog.payload())
        settings_service.save({"settings": settings})
    return MainWindow(settings_service=settings_service, settings=settings)


def run() -> None:
    QtCore.QLoggingCategory.setFilterRules("qt.accessibility.table.warning=false")
    app = QtWidgets.QApplication(sys.argv)
    _apply_palette(app)
    QtCore.QThreadPool.globalInstance().setMaxThreadCount(6)
    app.setApplicationName("semilabel_app")
    app.setOrganizationName("DamageDetector")
    smoke = os.environ.get("SEMILABEL_APP_SMOKE") == "1"
    window = build_window(interactive_connect=not smoke)
    if window is None:
        raise SystemExit(0)
    window.show()
    if smoke:
        QtCore.QTimer.singleShot(250, app.quit)
    raise SystemExit(app.exec())


def main() -> None:
    run()
