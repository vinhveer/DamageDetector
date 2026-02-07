from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoadedState:
    image_path: str
    image_w: int
    image_h: int
