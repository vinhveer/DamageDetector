"""Backward-compatible compatibility layer for the old monolithic SAM dataset module."""

from ..core import (
    DEFAULT_MASK_EXTS,
    build_mask_index,
    find_mask_path,
    get_mask_index,
    list_image_files,
    load_image_mask_arrays,
)
from .dataset import GenericDataset
from .prompts import build_prompt_tensors, random_rot_flip, random_rotate, sample_negative_point, select_point
from .transforms import RandomGenerator, RefineRandomGenerator, ValGenerator

__all__ = [
    "DEFAULT_MASK_EXTS",
    "GenericDataset",
    "RandomGenerator",
    "RefineRandomGenerator",
    "ValGenerator",
    "build_mask_index",
    "find_mask_path",
    "get_mask_index",
    "list_image_files",
    "load_image_mask_arrays",
    "build_prompt_tensors",
    "random_rot_flip",
    "random_rotate",
    "sample_negative_point",
    "select_point",
]
