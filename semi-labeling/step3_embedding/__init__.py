"""Step 3: persistent DINOv2 embeddings for Step 2 detections."""

from .cache_reader import load_embeddings

__all__ = ["load_embeddings"]
