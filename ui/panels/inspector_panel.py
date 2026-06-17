from __future__ import annotations

from PySide6 import QtCore, QtWidgets


def _make_form() -> QtWidgets.QFormLayout:
    form = QtWidgets.QFormLayout()
    form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
    form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
    form.setVerticalSpacing(6)
    form.setHorizontalSpacing(8)
    form.setContentsMargins(8, 8, 8, 8)
    return form


class _ImageTab(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        form = _make_form()
        self.path_label = QtWidgets.QLabel("—", self)
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.size_label = QtWidgets.QLabel("—", self)
        self.zoom_label = QtWidgets.QLabel("100%", self)
        form.addRow("File", self.path_label)
        form.addRow("Size", self.size_label)
        form.addRow("Zoom", self.zoom_label)
        outer.addLayout(form)
        outer.addStretch(1)


class _ObjectTab(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        form = _make_form()
        self.kind_label = QtWidgets.QLabel("—", self)
        self.label_edit = QtWidgets.QLineEdit(self)
        self.score_label = QtWidgets.QLabel("—", self)
        self.bbox_label = QtWidgets.QLabel("—", self)
        self.bbox_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("Type", self.kind_label)
        form.addRow("Label", self.label_edit)
        form.addRow("Score", self.score_label)
        form.addRow("Bbox", self.bbox_label)
        outer.addLayout(form)
        outer.addStretch(1)


class InspectorPanel(QtWidgets.QTabWidget):
    """Right dock top: Image info + Object properties (Detect/Segment are separate panels)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self.setDocumentMode(True)

        self.image_tab = _ImageTab(self)
        self.object_tab = _ObjectTab(self)
        self.addTab(self.image_tab, "Image")
        self.addTab(self.object_tab, "Object")
