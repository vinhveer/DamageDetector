"""Compatibility shim for the shared SAM backbone package.

The canonical implementation now lives under ``segmentation.sam.backbones``.
"""

from ...backbones.segment_anything import (
    SamPredictor,
    build_sam,
    build_sam_vit_b,
    build_sam_vit_h,
    build_sam_vit_l,
    sam_model_registry,
)

__all__ = [
    "build_sam",
    "build_sam_vit_h",
    "build_sam_vit_l",
    "build_sam_vit_b",
    "sam_model_registry",
    "SamPredictor",
]
