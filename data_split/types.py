from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    mask_path: Path | None
    stem: str
    source_id: str
    positive_ratio: float
