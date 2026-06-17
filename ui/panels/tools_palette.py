from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ui.canvas.tools import TOOL_REGISTRY

# Map tool name → Unicode symbol (fallback text label when no icon)
_TOOL_ICONS: dict[str, str] = {
    "pan": "✥",
    "select": "↖",
    "rect_roi": "▭",
    "polygon_roi": "⬡",
    "brush": "✏",
    "eraser": "⌫",
    "point_prompt": "⊕",
    "crop": "⊞",
    "measure": "↔",
}


class _ToolButton(QtWidgets.QToolButton):
    def __init__(self, name: str, label: str, shortcut: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedHeight(36)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

        symbol = _TOOL_ICONS.get(name, "•")
        text = f"  {symbol}  {label}"
        if shortcut:
            text += f"  [{shortcut}]"
        self.setText(text)

        self.setToolTip(f"{label}  ({shortcut})" if shortcut else label)
        self.setStyleSheet("""
            QToolButton {
                text-align: left;
                padding: 0 10px;
                font-size: 13px;
                border-radius: 5px;
            }
        """)


class ToolsPalette(QtWidgets.QWidget):
    """Left dock: vertical tool selector + tool options below."""

    toolSelected = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(130)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(2)

        self._buttons: dict[str, _ToolButton] = {}

        for name, cls in TOOL_REGISTRY.items():
            btn = _ToolButton(name, cls.label, cls.shortcut, self)
            btn.clicked.connect(lambda _=False, n=name: self.toolSelected.emit(n))
            self._buttons[name] = btn
            root.addWidget(btn)

        # Separator
        sep = QtWidgets.QFrame(self)
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        root.addWidget(sep)

        # Tool options host
        options_label = QtWidgets.QLabel("OPTIONS", self)
        options_label.setStyleSheet("color: rgba(180,180,180,0.5); font-size: 10px; font-weight: 700; letter-spacing: 1px;")
        options_label.setContentsMargins(4, 6, 0, 2)
        root.addWidget(options_label)

        self._options_host = QtWidgets.QWidget(self)
        self._options_host.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )
        self._options_layout = QtWidgets.QVBoxLayout(self._options_host)
        self._options_layout.setContentsMargins(0, 0, 0, 0)
        self._options_layout.setSpacing(4)
        root.addWidget(self._options_host, 1)

    def set_active(self, name: str) -> None:
        for btn_name, btn in self._buttons.items():
            btn.setChecked(btn_name == name)

    def set_options_widget(self, widget: QtWidgets.QWidget | None) -> None:
        while self._options_layout.count():
            item = self._options_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        if widget is not None:
            self._options_layout.addWidget(widget)
