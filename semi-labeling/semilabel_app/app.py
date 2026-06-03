from __future__ import annotations

import os
import sys

from PySide6 import QtCore, QtWidgets

from .config.defaults import DEFAULT_SETTINGS, migrate_settings
from .services.settings_service import SettingsService
from .ui.main_window import MainWindow


def build_window() -> MainWindow:
    settings_service = SettingsService()
    persisted = settings_service.load()
    settings = dict(DEFAULT_SETTINGS)
    settings.update(migrate_settings(dict(persisted.get("settings") or {})))
    return MainWindow(settings_service=settings_service, settings=settings)


def run() -> None:
    QtCore.QLoggingCategory.setFilterRules("qt.accessibility.table.warning=false")
    app = QtWidgets.QApplication(sys.argv)
    QtCore.QThreadPool.globalInstance().setMaxThreadCount(6)
    app.setApplicationName("semilabel_app")
    app.setOrganizationName("DamageDetector")
    window = build_window()
    window.show()
    if os.environ.get("SEMILABEL_APP_SMOKE") == "1":
        QtCore.QTimer.singleShot(250, app.quit)
    raise SystemExit(app.exec())


def main() -> None:
    run()