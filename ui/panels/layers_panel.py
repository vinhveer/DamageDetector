from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ui.models.layer import LayerKind, LayerNode, LayerTree

# Kind → small label
_KIND_BADGE: dict[LayerKind, str] = {
    LayerKind.image: "IMG",
    LayerKind.rois: "ROI",
    LayerKind.detections: "DET",
    LayerKind.masks: "MSK",
    LayerKind.measurements: "MSR",
    LayerKind.overlay: "OVR",
}

_KIND_COLOR: dict[LayerKind, str] = {
    LayerKind.image: "#888",
    LayerKind.rois: "#FFC629",
    LayerKind.detections: "#3796FF",
    LayerKind.masks: "#34D399",
    LayerKind.measurements: "#F59E0B",
    LayerKind.overlay: "#aaa",
}


class _LayerRow(QtWidgets.QWidget):
    """One row in the layers panel: eye | badge | name | opacity% | lock."""

    visibilityToggled = QtCore.Signal(str, bool)
    opacityChanged = QtCore.Signal(str, float)
    lockToggled = QtCore.Signal(str, bool)
    selected = QtCore.Signal(str)

    def __init__(self, layer: LayerNode, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._layer = layer
        self._updating = False
        self._active = False

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(4)

        # Eye toggle
        self._eye = QtWidgets.QToolButton(self)
        self._eye.setCheckable(True)
        self._eye.setChecked(layer.visible)
        self._eye.setFixedSize(20, 20)
        self._eye.setText("◉" if layer.visible else "○")
        self._eye.setToolTip("Toggle visibility")
        self._eye.setStyleSheet("QToolButton { border: none; font-size: 12px; padding: 0; }")
        self._eye.toggled.connect(self._on_eye)
        row.addWidget(self._eye)

        # Kind badge
        kind = layer.kind if isinstance(layer.kind, LayerKind) else LayerKind(layer.kind)
        badge = QtWidgets.QLabel(_KIND_BADGE.get(kind, "?"), self)
        badge.setFixedWidth(32)
        badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            f"color: {_KIND_COLOR.get(kind, '#aaa')}; font-size: 9px; font-weight: 700; "
            f"background: rgba(255,255,255,0.06); border-radius: 3px; padding: 1px 2px;"
        )
        row.addWidget(badge)

        # Name
        self._name = QtWidgets.QLabel(layer.name, self)
        self._name.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        font = self._name.font()
        if layer.locked:
            font.setItalic(True)
        self._name.setFont(font)
        row.addWidget(self._name, 1)

        # Opacity spinbox
        self._opacity = QtWidgets.QSpinBox(self)
        self._opacity.setRange(0, 100)
        self._opacity.setValue(int(layer.opacity * 100))
        self._opacity.setSuffix("%")
        self._opacity.setFixedWidth(58)
        self._opacity.setToolTip("Layer opacity")
        self._opacity.setEnabled(not layer.locked)
        self._opacity.valueChanged.connect(self._on_opacity)
        row.addWidget(self._opacity)

        # Lock toggle
        self._lock = QtWidgets.QToolButton(self)
        self._lock.setCheckable(True)
        self._lock.setChecked(layer.locked)
        self._lock.setFixedSize(20, 20)
        self._lock.setText("🔒" if layer.locked else "🔓")
        self._lock.setToolTip("Lock layer")
        self._lock.setStyleSheet("QToolButton { border: none; font-size: 11px; padding: 0; }")
        self._lock.toggled.connect(self._on_lock)
        row.addWidget(self._lock)

    def _on_eye(self, checked: bool) -> None:
        self._eye.setText("◉" if checked else "○")
        self.visibilityToggled.emit(self._layer.id, checked)

    def _on_opacity(self, value: int) -> None:
        self.opacityChanged.emit(self._layer.id, value / 100.0)

    def _on_lock(self, checked: bool) -> None:
        self._lock.setText("🔒" if checked else "🔓")
        self._opacity.setEnabled(not checked)
        font = self._name.font()
        font.setItalic(checked)
        self._name.setFont(font)
        self.lockToggled.emit(self._layer.id, checked)

    def refresh(self, layer: LayerNode) -> None:
        self._layer = layer
        self._updating = True
        self._eye.setChecked(layer.visible)
        self._eye.setText("◉" if layer.visible else "○")
        self._opacity.setValue(int(layer.opacity * 100))
        self._lock.setChecked(layer.locked)
        self._lock.setText("🔒" if layer.locked else "🔓")
        self._updating = False

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        self.setStyleSheet(
            "background: rgba(55,150,255,0.18); border-radius: 3px;" if active else ""
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self.selected.emit(self._layer.id)
        super().mousePressEvent(event)


class LayersPanel(QtWidgets.QWidget):
    """Right dock: layer list with per-row visibility / opacity / lock controls."""

    visibilityChanged = QtCore.Signal(str, bool)
    opacityChanged = QtCore.Signal(str, float)
    lockChanged = QtCore.Signal(str, bool)
    layerSelected = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self._active_layer_id: str | None = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Scroll area holding layer rows
        self._scroll = QtWidgets.QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._rows_widget = QtWidgets.QWidget(self._scroll)
        self._rows_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred
        )
        self._rows_layout = QtWidgets.QVBoxLayout(self._rows_widget)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(0)
        self._rows_layout.addStretch(1)
        self._scroll.setWidget(self._rows_widget)
        root.addWidget(self._scroll, 1)

        self._tree: LayerTree | None = None
        self._row_widgets: list[_LayerRow] = []

    def set_tree(self, tree: LayerTree) -> None:
        self._tree = tree
        self._populate()

    def _populate(self) -> None:
        # Remove old rows
        for w in self._row_widgets:
            w.setParent(None)
        self._row_widgets.clear()
        # Clear layout except the trailing stretch
        while self._rows_layout.count() > 1:
            self._rows_layout.takeAt(0)

        if self._tree is None:
            return

        # Build rows in reverse z_order (highest z = top of list = visually on top)
        layers = sorted(self._tree.layers(), key=lambda l: l.z_order, reverse=True)
        for layer in layers:
            row = _LayerRow(layer, self._rows_widget)
            row.visibilityToggled.connect(self._on_visibility)
            row.opacityChanged.connect(self._on_opacity)
            row.lockToggled.connect(self._on_lock)
            row.selected.connect(self._on_selected)
            row.set_active(layer.id == self._active_layer_id)

            # Alternating background via dynamic property
            sep = QtWidgets.QFrame(self._rows_widget)
            sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            sep.setStyleSheet("color: rgba(255,255,255,0.05);")

            self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)
            self._rows_layout.insertWidget(self._rows_layout.count() - 1, sep)
            self._row_widgets.append(row)

    def _on_visibility(self, layer_id: str, visible: bool) -> None:
        if self._tree is not None:
            self._tree.set_visible(layer_id, visible)
        self.visibilityChanged.emit(layer_id, visible)

    def _on_opacity(self, layer_id: str, opacity: float) -> None:
        if self._tree is not None:
            self._tree.set_opacity(layer_id, opacity)
        self.opacityChanged.emit(layer_id, opacity)

    def _on_lock(self, layer_id: str, locked: bool) -> None:
        if self._tree is not None:
            self._tree.set_locked(layer_id, locked)
        self.lockChanged.emit(layer_id, locked)

    def _on_selected(self, layer_id: str) -> None:
        self.set_active_layer(layer_id)
        self.layerSelected.emit(layer_id)

    def set_active_layer(self, layer_id: str | None) -> None:
        self._active_layer_id = layer_id
        for row in self._row_widgets:
            row.set_active(row._layer.id == layer_id)
