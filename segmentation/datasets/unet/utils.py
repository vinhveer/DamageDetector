import os

from PIL import ImageOps
from ..core.io import build_mask_index, find_mask_path, list_valid_images as _shared_list_valid_images



def _list_valid_images(image_dir, mask_dir, mask_prefix="auto", mask_index=None):
    return _shared_list_valid_images(
        image_dir,
        mask_dir,
        mask_prefix=mask_prefix,
        mask_index=mask_index,
    )


def _normalize_patch_size(patch_size):
    if patch_size is None:
        return None
    if isinstance(patch_size, int):
        return (patch_size, patch_size)
    if isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
        return (int(patch_size[0]), int(patch_size[1]))
    raise ValueError(f"Invalid patch_size: {patch_size!r}")


def _pad_to_min_size(img, min_w: int, min_h: int, fill):
    w, h = img.size
    pad_w = max(0, min_w - w)
    pad_h = max(0, min_h - h)
    if pad_w == 0 and pad_h == 0:
        return img

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    return ImageOps.expand(img, border=(left, top, right, bottom), fill=fill)
