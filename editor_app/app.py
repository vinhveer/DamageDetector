from __future__ import annotations

import sys

from PySide6 import QtWidgets

from .ui.main_window import MainWindow  # noqa: E402


def run() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())
