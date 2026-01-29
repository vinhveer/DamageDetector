import os

import numpy as np
import torch
from PIL import Image
from collections import OrderedDict
from torch.utils.data import Dataset

from .utils import _normalize_patch_size, _pad_to_min_size, build_mask_index, find_mask_path


class TiledDataset(Dataset):
    """
    Validation/Test dataset: deterministically tile each image into a full grid of patches.

    - stride = patch_size or patch_size//2 (overlap) to avoid cracks breaking on borders.
    - No random geometric augmentation.
    """

    def __init__(
        self,
        image_dir,
        mask_dir,
        mask_prefix="auto",
        patch_size=512,
        stride=None,
        image_transform=None,
        mask_transform=None,
        image_filenames=None,
        verbose=True,
        mask_index=None,
        cache_size=2,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_prefix = str(mask_prefix)
        self.patch_size = _normalize_patch_size(patch_size)
        self.stride = _normalize_patch_size(stride) if stride is not None else None
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.verbose = verbose
        self._cache_size = max(0, int(cache_size))
        self._cache = OrderedDict()

        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

        if mask_index is None:
            mask_index = build_mask_index(mask_dir)
        self._mask_index = mask_index

        if image_filenames is None:
            images = sorted(
                [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            )
            self.images = [
                img
                for img in images
                if find_mask_path(
                    mask_dir,
                    os.path.splitext(img)[0],
                    mask_prefix=self.mask_prefix,
                    mask_index=self._mask_index,
                )
                is not None
            ]
        else:
            self.images = list(image_filenames)

        if not self.images:
            raise RuntimeError("No valid image-mask pairs found.")

        patch_w, patch_h = self.patch_size
        if self.stride is None:
            self.stride = (patch_w // 2, patch_h // 2)
        stride_w, stride_h = self.stride
        if stride_w <= 0 or stride_h <= 0:
            raise ValueError("stride must be > 0")

        self.index = []
        for img_name in self.images:
            img_path = os.path.join(self.image_dir, img_name)
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception as e:
                if self.verbose:
                    print(f"TiledDataset: skipping unreadable image '{img_name}': {e}")
                continue

            w = max(w, patch_w)
            h = max(h, patch_h)

            xs = list(range(0, max(1, w - patch_w + 1), stride_w))
            ys = list(range(0, max(1, h - patch_h + 1), stride_h))
            if xs[-1] != w - patch_w:
                xs.append(w - patch_w)
            if ys[-1] != h - patch_h:
                ys.append(h - patch_h)

            for y in ys:
                for x in xs:
                    self.index.append((img_name, int(x), int(y)))

        if self.verbose:
            print(
                f"TiledDataset: {len(self.images)} image(s) -> {len(self.index)} patch(es), "
                f"patch_size={self.patch_size}, stride={self.stride}"
            )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        img_name, x, y = self.index[int(idx)]
        try:
            if self._cache_size > 0:
                cached = self._cache.get(img_name)
                if cached is not None:
                    self._cache.move_to_end(img_name)
                    image, mask = cached
                else:
                    image, mask = self._load_pair(img_name)
            else:
                image, mask = self._load_pair(img_name)

            patch_w, patch_h = self.patch_size
            image = _pad_to_min_size(image, patch_w, patch_h, fill=(0, 0, 0))
            mask = _pad_to_min_size(mask, patch_w, patch_h, fill=0)

            box = (x, y, x + patch_w, y + patch_h)
            image = image.crop(box)
            mask = mask.crop(box)

            if self.image_transform is not None:
                image = self.image_transform(image)
            else:
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
            else:
                mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0

            mask = (mask > 0.5).float()
            return image, mask
        except Exception as e:
            if self.verbose:
                print(f"TiledDataset error at idx={idx} ({img_name}): {e}")
            # Skip bad samples (collate_fn should drop None).
            return None

    def _load_pair(self, img_name):
        img_path = os.path.join(self.image_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        mask_path = find_mask_path(
            self.mask_dir,
            base_name,
            mask_prefix=self.mask_prefix,
            mask_index=self._mask_index,
        )
        if mask_path is None:
            raise FileNotFoundError(
                f"Mask not found for '{img_name}'. Expected '{base_name}{self.mask_prefix}.*' in {self.mask_dir}"
            )

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self._cache_size > 0:
            self._cache[img_name] = (image, mask)
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return image, mask
