from __future__ import annotations

import sys

from PySide6 import QtWidgets

from .main_window import MainWindow


def run() -> None:
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    raise SystemExit(app.exec())
