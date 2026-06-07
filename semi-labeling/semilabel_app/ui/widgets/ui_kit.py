from __future__ import annotations

from PySide6 import QtCore, QtWidgets


LABEL_BUTTON_STYLE: dict[str, tuple[str, str]] = {
    "crack": ("#4a90d9", "Crack [1]"),
    "mold": ("#27ae60", "Mold [2]"),
    "spall": ("#f39c12", "Spall [3]"),
    "reject": ("#e74c3c", "Reject [4]"),
}


def primary_button(text: str, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QPushButton:
    """Accent-filled call-to-action button."""
    btn = QtWidgets.QPushButton(text, parent)
    btn.setProperty("variant", "primary")
    btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
    btn.setMinimumHeight(32)
    return btn


def danger_button(text: str, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QPushButton:
    """Destructive-action button (red)."""
    btn = QtWidgets.QPushButton(text, parent)
    btn.setProperty("variant", "danger")
    btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
    btn.setMinimumHeight(32)
    return btn


def section_label(text: str, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QLabel:
    """Small uppercase muted heading."""
    label = QtWidgets.QLabel(text.upper(), parent)
    label.setObjectName("SectionLabel")
    return label


class Chip(QtWidgets.QLabel):
    """Small rounded badge for counts / status."""

    def __init__(self, text: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("Chip")
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)


class KeyboardHint(QtWidgets.QLabel):
    """Compact key hint used in action captions."""

    def __init__(self, text: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("KeyboardHint")
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(24)


class Toolbar(QtWidgets.QFrame):
    """Horizontal action bar that sits above page content."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Toolbar")
        self._layout = QtWidgets.QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)

    def add(self, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        self._layout.addWidget(widget)
        return widget

    def add_label(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text, self)
        label.setObjectName("ToolbarLabel")
        self._layout.addWidget(label)
        return label

    def add_stretch(self) -> None:
        self._layout.addStretch(1)

    def add_separator(self) -> None:
        line = QtWidgets.QFrame(self)
        line.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        line.setObjectName("ToolbarSep")
        self._layout.addWidget(line)


class Card(QtWidgets.QFrame):
    """Bordered surface with an optional title and a body layout."""

    def __init__(self, title: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self._outer = QtWidgets.QVBoxLayout(self)
        self._outer.setContentsMargins(14, 12, 14, 14)
        self._outer.setSpacing(10)
        if title:
            self._title = QtWidgets.QLabel(title, self)
            self._title.setObjectName("CardTitle")
            self._outer.addWidget(self._title)
        else:
            self._title = None
        self._body = QtWidgets.QVBoxLayout()
        self._body.setContentsMargins(0, 0, 0, 0)
        self._body.setSpacing(8)
        self._outer.addLayout(self._body, 1)

    def set_title(self, text: str) -> None:
        if self._title is not None:
            self._title.setText(text)

    def body(self) -> QtWidgets.QVBoxLayout:
        return self._body

    def add(self, widget: QtWidgets.QWidget, stretch: int = 0) -> QtWidgets.QWidget:
        self._body.addWidget(widget, stretch)
        return widget


class EmptyState(QtWidgets.QFrame):
    """Centered hint shown when a view has no data yet."""

    def __init__(self, text: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("EmptyState")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._label = QtWidgets.QLabel(text, self)
        self._label.setObjectName("EmptyStateText")
        self._label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)
        layout.addWidget(self._label)

    def set_text(self, text: str) -> None:
        self._label.setText(text)


class InfoPanel(QtWidgets.QFrame):
    """Key / value rows replacing raw text dumps."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("InfoPanel")
        self._grid = QtWidgets.QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(6)
        self._grid.setColumnStretch(1, 1)

    def set_rows(self, rows: list[tuple[str, str]]) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for row, (key, value) in enumerate(rows):
            key_label = QtWidgets.QLabel(str(key), self)
            key_label.setObjectName("InfoKey")
            val_label = QtWidgets.QLabel(str(value), self)
            val_label.setObjectName("InfoVal")
            val_label.setWordWrap(True)
            val_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            self._grid.addWidget(key_label, row, 0, QtCore.Qt.AlignmentFlag.AlignTop)
            self._grid.addWidget(val_label, row, 1)

    def clear(self) -> None:
        self.set_rows([])


class PickedList(QtWidgets.QListWidget):
    """List of currently picked items; emits the result_id to un-pick."""

    removeRequested = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PickedList")
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.setAlternatingRowColors(True)
        self.itemClicked.connect(self._on_clicked)

    def set_entries(self, entries: list[tuple[int, str]]) -> None:
        self.clear()
        for result_id, text in entries:
            item = QtWidgets.QListWidgetItem(text, self)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, int(result_id))
            item.setToolTip("Bấm để bỏ chọn")
            self.addItem(item)

    def _on_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        self.removeRequested.emit(int(item.data(QtCore.Qt.ItemDataRole.UserRole)))


class DecisionBar(QtWidgets.QFrame):
    """Horizontal decision/action strip that emits action ids."""

    decided = QtCore.Signal(str)

    def __init__(
        self,
        actions: list[tuple[str, str, str]],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("DecisionBar")
        self._buttons: dict[str, QtWidgets.QPushButton] = {}
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)
        self._row = QtWidgets.QHBoxLayout()
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(8)
        root.addLayout(self._row)
        self.caption = QtWidgets.QLabel("", self)
        self.caption.setObjectName("DecisionCaption")
        self.caption.setWordWrap(True)
        root.addWidget(self.caption)
        for action_id, text, color in actions:
            button = QtWidgets.QPushButton(text, self)
            button.setCheckable(True)
            button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            button.setMinimumHeight(34)
            button.setProperty("decisionColor", color)
            button.setStyleSheet(
                "QPushButton {"
                f"background: {color}22; border: 1px solid {color}; color: #1f2a33;"
                "font-weight: 600; padding: 6px 10px; border-radius: 6px;"
                "}"
                "QPushButton:hover { background: rgba(255, 255, 255, 0.75); }"
                "QPushButton:checked {"
                f"background: {color}; color: white; border-color: {color};"
                "}"
            )
            button.clicked.connect(lambda _checked=False, value=action_id: self.decided.emit(value))
            self._buttons[action_id] = button
            self._row.addWidget(button)
        self._row.addStretch(1)

    def set_current(self, action_id: str) -> None:
        for key, button in self._buttons.items():
            button.setChecked(bool(action_id) and key == action_id)

    def set_button_text(self, action_id: str, text: str) -> None:
        button = self._buttons.get(action_id)
        if button is not None:
            button.setText(text)

    def set_caption(self, text: str) -> None:
        self.caption.setText(str(text or ""))


class PercentBar(QtWidgets.QWidget):
    """Table cell widget: a label with a proportional background bar."""

    def __init__(self, fraction: float, text: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._fraction = max(0.0, min(1.0, float(fraction)))
        self._text = str(text)
        self.setMinimumHeight(22)

    def set_value(self, fraction: float, text: str) -> None:
        self._fraction = max(0.0, min(1.0, float(fraction)))
        self._text = str(text)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: ANN001
        from PySide6 import QtGui

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(2, 3, -2, -3)
        radius = 4
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor("#e8eaec"))
        painter.drawRoundedRect(rect, radius, radius)
        fill = QtCore.QRectF(rect)
        fill.setWidth(rect.width() * self._fraction)
        painter.setBrush(QtGui.QColor("#3daee9"))
        painter.drawRoundedRect(fill, radius, radius)
        painter.setPen(QtGui.QColor("#232629"))
        painter.drawText(rect.adjusted(8, 0, -8, 0), QtCore.Qt.AlignmentFlag.AlignVCenter, self._text)
