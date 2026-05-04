from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Roi:
    id: int
    image_rel_path: str
    name: str
    x: int
    y: int
    size: int

