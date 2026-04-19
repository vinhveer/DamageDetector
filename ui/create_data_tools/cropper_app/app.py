from __future__ import annotations

import sys

from PySide6 import QtCore, QtWidgets

from .ui.main_window import MainWindow


def run() -> None:
    QtCore.QLoggingCategory.setFilterRules("qt.accessibility.table.warning=false")
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())
