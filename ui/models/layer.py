from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable
from uuid import uuid4


class LayerKind(str, Enum):
    image = "image"
    rois = "rois"
    detections = "detections"
    masks = "masks"
    measurements = "measurements"
    overlay = "overlay"


@dataclass
class MaskRef:
    """A segmentation mask produced for a detection group."""

    mask_path: str
    box: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])


@dataclass
class DetectionGroup:
    """One detection run: its boxes (rows) plus masks segmented from them.

    `layer_id` ties the group to a dynamic LayerNode of kind `detections` so
    visibility/opacity controls in the Layers panel act on this group.
    """

    layer_id: str
    name: str
    detector: str
    rows: list[Any] = field(default_factory=list)
    mask_refs: list[MaskRef] = field(default_factory=list)


@dataclass
class LayerNode:
    id: str = field(default_factory=lambda: f"L_{uuid4().hex[:8]}")
    kind: LayerKind = LayerKind.overlay
    name: str = ""
    visible: bool = True
    locked: bool = False
    opacity: float = 1.0
    z_order: int = 0
    item_ids: list[str] = field(default_factory=list)


def default_layers() -> list[LayerNode]:
    # Detection-group layers are created dynamically (one per detect run); only
    # the static, always-present layers live here.
    return [
        LayerNode(kind=LayerKind.image, name="Image", locked=True, z_order=0),
        LayerNode(kind=LayerKind.rois, name="ROIs", z_order=10),
        LayerNode(kind=LayerKind.measurements, name="Measurements", visible=False, z_order=40),
    ]


class LayerTree:
    def __init__(self, layers: Iterable[LayerNode] | None = None) -> None:
        self._layers: list[LayerNode] = list(layers) if layers is not None else default_layers()

    def layers(self) -> list[LayerNode]:
        return list(self._layers)

    def by_kind(self, kind: LayerKind) -> LayerNode | None:
        for layer in self._layers:
            if layer.kind == kind:
                return layer
        return None

    def by_id(self, layer_id: str) -> LayerNode | None:
        for layer in self._layers:
            if layer.id == layer_id:
                return layer
        return None

    def set_visible(self, layer_id: str, visible: bool) -> None:
        layer = self.by_id(layer_id)
        if layer is not None:
            layer.visible = bool(visible)

    def set_opacity(self, layer_id: str, opacity: float) -> None:
        layer = self.by_id(layer_id)
        if layer is not None:
            layer.opacity = float(max(0.0, min(1.0, opacity)))

    def set_locked(self, layer_id: str, locked: bool) -> None:
        layer = self.by_id(layer_id)
        if layer is not None:
            layer.locked = bool(locked)

    def add_layer(self, node: LayerNode) -> None:
        self._layers.append(node)

    def remove_layer(self, layer_id: str) -> None:
        self._layers = [layer for layer in self._layers if layer.id != layer_id]

    def detection_layers(self) -> list[LayerNode]:
        return [layer for layer in self._layers if layer.kind == LayerKind.detections]
