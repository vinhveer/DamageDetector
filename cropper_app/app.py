from __future__ import annotations

import sys
from pathlib import Path

from PySide6 import QtCore, QtWidgets

try:
    from cropper_app.ui.main_window import MainWindow
except ModuleNotFoundError as exc:
    if getattr(exc, "name", "") != "cropper_app":
        raise
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from cropper_app.ui.main_window import MainWindow


def run() -> None:
    QtCore.QLoggingCategory.setFilterRules("qt.accessibility.table.warning=false")
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())
