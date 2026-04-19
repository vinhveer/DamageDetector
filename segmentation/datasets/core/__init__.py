"""Shared dataset core for segmentation pipelines."""

from .augment import (
    build_crack_profile_augment,
    build_imagenet_normalize,
)
from .crop import (
    build_crop_metadata,
    choose_centered_foreground_crop_coords,
    choose_refine_crop_coords,
    choose_smart_crack_crop_coords,
    hard_negative_crop_coords,
    random_crop_coords,
)
from .io import (
    DEFAULT_IMAGE_EXTS,
    DEFAULT_MASK_EXTS,
    PairRecord,
    build_mask_index,
    find_mask_path,
    get_mask_index,
    list_image_files,
    list_valid_images,
    list_valid_pairs,
    load_image_mask_arrays,
)
from .sample import (
    crop_image_label,
    ensure_numpy_image_label,
    ensure_three_channel_image,
    normalize_binary_mask,
    pad_canvas_if_needed,
    resize_with_center_padding,
    to_uint8_image,
)

__all__ = [
    "DEFAULT_IMAGE_EXTS",
    "DEFAULT_MASK_EXTS",
    "PairRecord",
    "build_crack_profile_augment",
    "build_crop_metadata",
    "build_imagenet_normalize",
    "build_mask_index",
    "choose_centered_foreground_crop_coords",
    "choose_refine_crop_coords",
    "choose_smart_crack_crop_coords",
    "crop_image_label",
    "ensure_numpy_image_label",
    "ensure_three_channel_image",
    "find_mask_path",
    "get_mask_index",
    "hard_negative_crop_coords",
    "list_image_files",
    "list_valid_images",
    "list_valid_pairs",
    "load_image_mask_arrays",
    "normalize_binary_mask",
    "pad_canvas_if_needed",
    "random_crop_coords",
    "resize_with_center_padding",
    "to_uint8_image",
]
