from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, float(self.x2) - float(self.x1))

    @property
    def height(self) -> float:
        return max(0.0, float(self.y2) - float(self.y1))

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_xyxy(self) -> list[float]:
        return [float(self.x1), float(self.y1), float(self.x2), float(self.y2)]

    def as_int_xyxy(self) -> tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2)))


@dataclass(frozen=True)
class ImageInfo:
    path: Path
    rel_path: str
    width: int
    height: int


@dataclass(frozen=True)
class Tile:
    index: int
    box: Box


@dataclass
class Detection:
    box: Box
    label: str
    score: float
    prompt_key: str
    prompt_text: str
    stage: str
    source: str
    model_name: str = "groundingdino"
    parent_detection_id: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def box_w(self) -> int:
        return int(round(self.box.width))

    @property
    def box_h(self) -> int:
        return int(round(self.box.height))

    @property
    def area_px2(self) -> int:
        return int(round(self.box.area))
