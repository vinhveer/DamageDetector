"""Generic options dialog used to move per-page settings out of the toolbars.

Pages declare their options as a small spec list; the dialog renders the right
editor (choice combo or integer spin) and returns the chosen values.  This keeps
the main views free of inline controls — the user opens options only when they
want to change something.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PySide6 import QtWidgets
from PySide6.QtCore import Qt


@dataclass
class Option:
    key: str
    label: str
    kind: str  # "choice" | "int"
    value: Any
    choices: list[str] = field(default_factory=list)
    minimum: int = 0
    maximum: int = 100000
    step: int = 50


class OptionsDialog(QtWidgets.QDialog):
    def __init__(self, title: str, options: list[Option], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(360)
        self._editors: dict[str, QtWidgets.QWidget] = {}

        root = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        for opt in options:
            editor = self._make_editor(opt)
            self._editors[opt.key] = editor
            form.addRow(opt.label, editor)
        root.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _make_editor(self, opt: Option) -> QtWidgets.QWidget:
        if opt.kind == "choice":
            combo = QtWidgets.QComboBox(self)
            combo.addItems([str(c) for c in opt.choices])
            idx = combo.findText(str(opt.value))
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            return combo
        spin = QtWidgets.QSpinBox(self)
        spin.setRange(int(opt.minimum), int(opt.maximum))
        spin.setSingleStep(int(opt.step))
        spin.setValue(int(opt.value))
        return spin

    def values(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, editor in self._editors.items():
            if isinstance(editor, QtWidgets.QComboBox):
                out[key] = editor.currentText()
            elif isinstance(editor, QtWidgets.QSpinBox):
                out[key] = editor.value()
        return out

    @staticmethod
    def edit(title: str, options: list[Option], parent: QtWidgets.QWidget | None = None) -> dict[str, Any] | None:
        dialog = OptionsDialog(title, options, parent)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return dialog.values()
