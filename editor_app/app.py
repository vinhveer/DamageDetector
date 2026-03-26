from __future__ import annotations

import sys

from PySide6 import QtCore, QtWidgets

from .ui.main_window import MainWindow  # noqa: E402


def run() -> None:
    # Qt occasionally emits noisy accessibility table warnings while models refresh.
    # They are harmless for this app and drown out useful logs.
    QtCore.QLoggingCategory.setFilterRules("qt.accessibility.table.warning=false")
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())
