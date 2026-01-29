import os
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance
from collections import OrderedDict
from torch.utils.data import Dataset

from .utils import _normalize_patch_size, _pad_to_min_size, build_mask_index, find_mask_path


class RandomPatchDataset(Dataset):
    """
    Training dataset: each item is a random crop patch from an image.

    - __len__ = num_images * patches_per_image
    - __getitem__ maps an index to an image, then samples a random patch.
    - Tries K times to find a patch containing crack pixels; falls back to random.
    """

    def __init__(
        self,
        image_dir,
        mask_dir,
        mask_prefix="auto",
        patch_size=512,
        patches_per_image=2,
        max_patch_tries=5,
        augment=False,
        image_transform=None,
        mask_transform=None,
        image_filenames=None,
        verbose=True,
        p_rotate=0.5,
        p_hflip=0.5,
        p_vflip=0.3,
        p_brightness=0.3,
        p_contrast=0.3,
        mask_index=None,
        cache_size=4,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_prefix = str(mask_prefix)
        self.patch_size = _normalize_patch_size(patch_size)
        self.patches_per_image = int(patches_per_image)
        self.max_patch_tries = int(max_patch_tries)
        self.augment = augment
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.verbose = verbose
        self.p_rotate = float(p_rotate)
        self.p_hflip = float(p_hflip)
        self.p_vflip = float(p_vflip)
        self.p_brightness = float(p_brightness)
        self.p_contrast = float(p_contrast)
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

        if self.verbose:
            print(f"RandomPatchDataset: {len(self.images)} image(s), patches_per_image={self.patches_per_image}")

    def __len__(self):
        return len(self.images) * self.patches_per_image

    def _load_pair(self, img_name):
        if self._cache_size > 0:
            cached = self._cache.get(img_name)
            if cached is not None:
                self._cache.move_to_end(img_name)
                return cached

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

    def _random_patch(self, image: Image.Image, mask: Image.Image):
        patch_w, patch_h = self.patch_size
        image = _pad_to_min_size(image, patch_w, patch_h, fill=(0, 0, 0))
        mask = _pad_to_min_size(mask, patch_w, patch_h, fill=0)

        w, h = image.size
        left = 0 if w == patch_w else random.randint(0, w - patch_w)
        top = 0 if h == patch_h else random.randint(0, h - patch_h)
        box = (left, top, left + patch_w, top + patch_h)
        return image.crop(box), mask.crop(box)

    def __getitem__(self, idx):
        img_idx = int(idx) // self.patches_per_image
        img_idx = max(0, min(img_idx, len(self.images) - 1))
        img_name = self.images[img_idx]

        try:
            image, mask = self._load_pair(img_name)

            # 1) Sample patch first (try to get crack patch).
            best = None
            for _ in range(max(1, self.max_patch_tries)):
                img_p, mask_p = self._random_patch(image, mask)
                best = (img_p, mask_p)
                if mask_p.getbbox() is not None:
                    break

            image, mask = best

            # 2) Augment on patch (cheaper than full image).
            if self.augment:
                if random.random() < self.p_rotate:
                    rot = random.choice([Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
                    image = image.transpose(rot)
                    mask = mask.transpose(rot)

                if random.random() < self.p_hflip:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

                if random.random() < self.p_vflip:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                    mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

                if random.random() < self.p_brightness:
                    enhancer = ImageEnhance.Brightness(image)
                    image = enhancer.enhance(random.uniform(0.8, 1.2))

                if random.random() < self.p_contrast:
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(random.uniform(0.8, 1.2))

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
                print(f"RandomPatchDataset error at idx={idx} ({img_name}): {e}")
            # Skip bad samples (collate_fn should drop None).
            return None
