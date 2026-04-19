import os
from dataclasses import dataclass


DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
DEFAULT_MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_MASK_INDEX_CACHE: dict[tuple[str, tuple[str, ...]], dict[str, str]] = {}


@dataclass(frozen=True)
class PairRecord:
    image_name: str
    base_name: str
    image_path: str
    mask_path: str


def _normalize_exts(exts, default_exts):
    return tuple(str(ext).lower() for ext in (exts or default_exts))


def build_mask_index(mask_dir: str, mask_exts=None) -> dict[str, str]:
    ext_values = _normalize_exts(mask_exts, DEFAULT_MASK_EXTS)
    index: dict[str, str] = {}
    if not os.path.isdir(mask_dir):
        return index

    def _sort_key(name: str) -> tuple[int, str]:
        ext = os.path.splitext(name)[1].lower()
        try:
            ext_rank = ext_values.index(ext)
        except ValueError:
            ext_rank = len(ext_values)
        return ext_rank, name

    for file_name in sorted(os.listdir(mask_dir), key=_sort_key):
        stem, ext = os.path.splitext(file_name)
        if ext.lower() not in ext_values:
            continue
        full_path = os.path.join(mask_dir, file_name)
        if not os.path.isfile(full_path):
            continue
        if stem and stem not in index:
            index[stem] = full_path
    return index


def get_mask_index(mask_dir: str, mask_exts=None) -> dict[str, str]:
    ext_values = _normalize_exts(mask_exts, DEFAULT_MASK_EXTS)
    cache_key = (os.path.abspath(mask_dir), ext_values)
    mask_index = _MASK_INDEX_CACHE.get(cache_key)
    if mask_index is None:
        mask_index = build_mask_index(mask_dir, mask_exts=ext_values)
        _MASK_INDEX_CACHE[cache_key] = mask_index
    return mask_index


def find_mask_path(
    mask_dir: str,
    image_base_name: str,
    mask_prefix: str = "auto",
    *,
    mask_index=None,
    mask_exts=None,
):
    if mask_prefix is None:
        prefixes = ["_mask", ""]
    else:
        prefix_value = str(mask_prefix)
        prefixes = ["_mask", ""] if prefix_value.lower() == "auto" else [prefix_value]

    index = mask_index if mask_index is not None else get_mask_index(mask_dir, mask_exts=mask_exts)
    for prefix in prefixes:
        candidate = index.get(f"{image_base_name}{prefix}")
        if candidate:
            return candidate
    return None


def list_image_files(image_dir: str, img_exts=None) -> list[str]:
    ext_values = _normalize_exts(img_exts, DEFAULT_IMAGE_EXTS)
    names: list[str] = []
    for file_name in os.listdir(image_dir):
        if os.path.splitext(file_name)[1].lower() in ext_values:
            names.append(file_name)
    names.sort()
    return names


def list_valid_pairs(
    image_dir: str,
    mask_dir: str,
    *,
    mask_prefix: str = "auto",
    image_filenames=None,
    mask_index=None,
    img_exts=None,
    mask_exts=None,
) -> list[PairRecord]:
    ext_values = _normalize_exts(img_exts, DEFAULT_IMAGE_EXTS)
    index = mask_index if mask_index is not None else get_mask_index(mask_dir, mask_exts=mask_exts)
    if image_filenames is None:
        image_names = list_image_files(image_dir, img_exts=ext_values)
    else:
        image_names = list(image_filenames)

    pairs: list[PairRecord] = []
    for image_name in image_names:
        if os.path.splitext(image_name)[1].lower() not in ext_values:
            continue
        base_name = os.path.splitext(image_name)[0]
        mask_path = find_mask_path(
            mask_dir,
            base_name,
            mask_prefix=mask_prefix,
            mask_index=index,
            mask_exts=mask_exts,
        )
        if mask_path is None:
            continue
        pairs.append(
            PairRecord(
                image_name=image_name,
                base_name=base_name,
                image_path=os.path.join(image_dir, image_name),
                mask_path=mask_path,
            )
        )
    return pairs


def list_valid_images(
    image_dir: str,
    mask_dir: str,
    *,
    mask_prefix: str = "auto",
    image_filenames=None,
    mask_index=None,
    img_exts=None,
    mask_exts=None,
) -> list[str]:
    return [
        record.image_name
        for record in list_valid_pairs(
            image_dir,
            mask_dir,
            mask_prefix=mask_prefix,
            image_filenames=image_filenames,
            mask_index=mask_index,
            img_exts=img_exts,
            mask_exts=mask_exts,
        )
    ]


def load_image_mask_arrays(base_dir: str, image_name: str, *, mask_prefix: str = "auto", img_exts=None, mask_exts=None):
    from PIL import Image
    import numpy as np

    img_dir = os.path.join(base_dir, "images")
    mask_dir = os.path.join(base_dir, "masks")
    image_path = os.path.join(img_dir, image_name)
    base_name = os.path.splitext(image_name)[0]
    mask_path = find_mask_path(
        mask_dir,
        base_name,
        mask_prefix=mask_prefix,
        mask_index=get_mask_index(mask_dir, mask_exts=mask_exts),
        mask_exts=mask_exts,
    )
    if mask_path is None:
        raise FileNotFoundError(f"No corresponding mask found for image {image_name} in {mask_dir}")
    image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    mask = (np.array(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0).astype(np.uint8)
    return image, mask
