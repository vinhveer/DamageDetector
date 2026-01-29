import os

from PIL import ImageOps

_PREFERRED_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


def _list_valid_images(image_dir, mask_dir, mask_prefix="auto", mask_index=None):
    if mask_index is None:
        mask_index = build_mask_index(mask_dir)
    images = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    )
    valid = []
    for img in images:
        base_name = os.path.splitext(img)[0]
        mask_path = find_mask_path(
            mask_dir, base_name, mask_prefix=mask_prefix, mask_index=mask_index
        )
        if mask_path is not None:
            valid.append(img)
    return valid


def build_mask_index(mask_dir: str) -> dict:
    try:
        names = os.listdir(mask_dir)
    except FileNotFoundError:
        return {}

    by_stem = {}
    for name in names:
        stem, _ext = os.path.splitext(name)
        if not stem:
            continue
        by_stem.setdefault(stem, []).append(name)

    best = {}
    for stem, stem_names in by_stem.items():
        candidates_sorted = sorted(stem_names)
        picked = None
        for ext in _PREFERRED_EXTS:
            for name in candidates_sorted:
                if os.path.splitext(name)[1].lower() == ext:
                    picked = name
                    break
            if picked is not None:
                break
        if picked is None:
            picked = candidates_sorted[0]
        best[stem] = os.path.join(mask_dir, picked)
    return best


def find_mask_path(
    mask_dir: str,
    image_base_name: str,
    mask_prefix: str = "auto",
    *,
    mask_index=None,
):
    """
    Find a mask file by base name + optional suffix, allowing any extension.

    Example:
      image 'abc.jpg' -> image_base_name='abc'
      mask_prefix='_mask' => look for 'abc_mask.*'
      mask_prefix='' => look for 'abc.*'
      mask_prefix='auto' => try both '' and '_mask'
    """
    if mask_prefix is None:
        prefixes = ["_mask", ""]
    else:
        mask_prefix = str(mask_prefix)
        if mask_prefix.lower() == "auto":
            prefixes = ["_mask", ""]
        else:
            prefixes = [mask_prefix]

    if mask_index is None:
        mask_index = build_mask_index(mask_dir)
    for prefix in prefixes:
        target_stem = f"{image_base_name}{prefix}"
        candidate = mask_index.get(target_stem)
        if candidate:
            return candidate
    return None


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
