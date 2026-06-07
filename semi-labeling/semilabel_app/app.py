from __future__ import annotations

import os
import sys

from PySide6 import QtCore, QtWidgets

from .config.defaults import DEFAULT_SETTINGS, migrate_settings
from .services.settings_service import SettingsService
from .ui.main_window import ConnectDialog, MainWindow


_APP_STYLE = """
QMainWindow, QWidget {
    font-size: 12px;
}
QFrame#ConnectionBar {
    background: #f2f4f6;
    border-bottom: 1px solid #d8dde3;
}
QFrame#Toolbar {
    background: transparent;
}
QLabel#ToolbarLabel, QLabel#InfoKey {
    color: #68717a;
    font-weight: 600;
}
QLabel#Chip {
    background: #eef2f5;
    border: 1px solid #d6dde5;
    border-radius: 8px;
    padding: 3px 8px;
    color: #2d3842;
}
QFrame#Card {
    background: #ffffff;
    border: 1px solid #dce1e6;
    border-radius: 8px;
}
QLabel#CardTitle {
    color: #2d3842;
    font-weight: 700;
}
QLabel#InfoVal {
    color: #20262d;
}
QFrame#DecisionBar {
    background: #f8f9fa;
    border: 1px solid #dce1e6;
    border-radius: 8px;
}
QLabel#DecisionCaption {
    color: #68717a;
}
QPushButton {
    min-height: 28px;
    padding: 5px 10px;
}
QPushButton[variant="primary"] {
    background: #2563eb;
    border: 1px solid #1d4ed8;
    color: white;
    border-radius: 6px;
    font-weight: 600;
}
QPushButton[variant="danger"] {
    background: #fee2e2;
    border: 1px solid #ef4444;
    color: #991b1b;
    border-radius: 6px;
    font-weight: 600;
}
QListView, QListWidget {
    background: #ffffff;
    border: 1px solid #dce1e6;
    border-radius: 8px;
}
"""


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
    app.setStyleSheet(_APP_STYLE)
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
