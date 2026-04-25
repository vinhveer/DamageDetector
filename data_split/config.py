from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from object_detection.dinov2.dinov2_prototypes import default_dinov2_embedding_checkpoint


@dataclass(slots=True)
class SplitConfig:
    input_root: Path
    output_root: Path
    splits: tuple[float, float, float] = (0.7, 0.15, 0.15)
    split_names: tuple[str, str, str] = ("train", "val", "test")
    num_clusters: int = 36
    batch_size: int = 16
    device: str = "auto"
    checkpoint: str = field(default_factory=default_dinov2_embedding_checkpoint)
    mask_threshold: int = 127
