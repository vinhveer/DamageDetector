try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: PySide6. Install it with:\n"
        "  pip install -r requirements.txt\n"
        "or:\n"
        "  pip install PySide6==6.7.3"
    ) from e

__all__ = ["QtCore", "QtGui", "QtWidgets"]
