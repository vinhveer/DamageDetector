from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Category:
    id: int
    name: str


@dataclass(frozen=True)
class CocoSplit:
    name: str
    annotation_path: Path
    image_dir: Path


@dataclass(frozen=True)
class SemiDataset:
    root: Path
    categories: tuple[Category, ...]
    splits: tuple[CocoSplit, ...]
