from __future__ import annotations

import sys

from PySide6 import QtCore, QtGui, QtWidgets

from ui.app.main_window import MainWindow


APP_NAME = "DamageDetector"


def _setup_app(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("DamageDetector")

    # Crisper font rendering on macOS/Linux
    font = app.font()
    font.setPointSize(13)
    app.setFont(font)

    # Dark Fusion palette — clean neutral dark theme
    palette = QtGui.QPalette()
    bg = QtGui.QColor(38, 38, 38)
    mid = QtGui.QColor(48, 48, 48)
    alt = QtGui.QColor(44, 44, 44)
    text = QtGui.QColor(220, 220, 220)
    dim = QtGui.QColor(140, 140, 140)
    accent = QtGui.QColor(55, 140, 255)
    hi_text = QtGui.QColor(255, 255, 255)

    palette.setColor(QtGui.QPalette.ColorRole.Window, bg)
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, text)
    palette.setColor(QtGui.QPalette.ColorRole.Base, mid)
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, alt)
    palette.setColor(QtGui.QPalette.ColorRole.Text, text)
    palette.setColor(QtGui.QPalette.ColorRole.BrightText, hi_text)
    palette.setColor(QtGui.QPalette.ColorRole.Button, mid)
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, text)
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, accent)
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, hi_text)
    palette.setColor(QtGui.QPalette.ColorRole.PlaceholderText, dim)
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(60, 60, 60))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.WindowText, dim)
    palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Text, dim)
    palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.ButtonText, dim)
    app.setPalette(palette)

    # Minimal stylesheet for things Fusion palette alone can't fix
    app.setStyleSheet("""
        QDockWidget::title {
            padding: 4px 8px;
            font-weight: 600;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        QTabBar::tab {
            padding: 5px 12px;
            min-width: 60px;
        }
        QTabBar::tab:selected {
            font-weight: 600;
        }
        QToolButton {
            border: 1px solid transparent;
            border-radius: 4px;
            padding: 4px 8px;
        }
        QToolButton:hover {
            border-color: rgba(255,255,255,0.12);
        }
        QToolButton:checked {
            background: rgba(55,140,255,0.25);
            border-color: rgba(55,140,255,0.6);
        }
        QSplitter::handle {
            background: rgba(255,255,255,0.06);
        }
        QSplitter::handle:horizontal { width: 1px; }
        QSplitter::handle:vertical   { height: 1px; }
        QHeaderView::section {
            padding: 4px 6px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        QTableWidget { gridline-color: rgba(255,255,255,0.06); }
        QPushButton {
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 4px;
            padding: 4px 12px;
            min-height: 22px;
        }
        QPushButton:hover { border-color: rgba(255,255,255,0.22); }
        QPushButton:pressed { background: rgba(0,0,0,0.3); }
        QPushButton:disabled { color: #666; border-color: rgba(255,255,255,0.05); }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 3px;
            padding: 2px 6px;
            min-height: 22px;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border-color: rgba(55,140,255,0.7);
        }
        QScrollBar:vertical { width: 8px; }
        QScrollBar:horizontal { height: 8px; }
        QScrollBar::handle { border-radius: 4px; background: rgba(255,255,255,0.18); }
        QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
        QStatusBar { font-size: 12px; }
        QStatusBar::item { border: none; padding: 0 4px; }
    """)


def create_app(argv: list[str] | None = None) -> QtWidgets.QApplication:
    QtWidgets.QApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    app = QtWidgets.QApplication(sys.argv if argv is None else argv)
    _setup_app(app)
    return app


def run(argv: list[str] | None = None) -> int:
    app = create_app(argv)
    window = MainWindow()
    window.show()
    return app.exec()
