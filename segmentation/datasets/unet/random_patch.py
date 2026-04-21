import os
import random

import numpy as np
from torch_runtime import torch
from PIL import Image, ImageEnhance
from collections import OrderedDict
from torch_runtime import Dataset

from ..core import (
    build_crop_metadata,
    build_crack_profile_augment,
    choose_smart_crack_crop_coords,
    crop_image_label,
    ensure_three_channel_image,
    list_valid_pairs,
    normalize_binary_mask,
    pad_canvas_if_needed,
    to_uint8_image,
)
from .utils import _normalize_patch_size, _pad_to_min_size, build_mask_index, find_mask_path
import albumentations as A



class RandomPatchDataset(Dataset):
    """
    Training dataset: each item is a random crop patch from an image.
    Uses Albumentations for fast augmentation and cropping.
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
        aug_prob=0.5,
        rotate_limit=30.0,
        brightness_limit=0.2,
        contrast_limit=0.2,
        negative_patch_prob=0.25,
        mask_index=None,
        cache_size=4,
        augment_profile="balanced",
        crop_policy="smart",
        background_crop_prob=0.2,
        near_background_crop_prob=0.15,
        hard_negative_crop_prob=0.1,
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
        self.aug_prob = max(0.0, min(1.0, float(aug_prob)))
        self.rotate_limit = float(rotate_limit)
        self.brightness_limit = float(brightness_limit)
        self.contrast_limit = float(contrast_limit)
        self.negative_patch_prob = max(0.0, min(1.0, float(negative_patch_prob)))
        self.augment_profile = str(augment_profile or "balanced").strip().lower()
        self.crop_policy = str(crop_policy or "smart").strip().lower()
        self.background_crop_prob = float(background_crop_prob if background_crop_prob is not None else self.negative_patch_prob)
        self.near_background_crop_prob = float(near_background_crop_prob)
        self.hard_negative_crop_prob = float(hard_negative_crop_prob)
        self._cache_size = max(0, int(cache_size))
        self._cache = OrderedDict()

        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

        if mask_index is None:
            mask_index = build_mask_index(mask_dir)
        self._mask_index = mask_index

        self._records = list_valid_pairs(
            image_dir,
            mask_dir,
            mask_prefix=self.mask_prefix,
            image_filenames=image_filenames,
            mask_index=self._mask_index,
        )
        self._record_by_name = {record.image_name: record for record in self._records}
        self.images = [record.image_name for record in self._records]

        if not self.images:
            raise RuntimeError("No valid image-mask pairs found.")

        if self.verbose:
            print(f"RandomPatchDataset: {len(self.images)} image(s), patches_per_image={self.patches_per_image}")

        patch_h, patch_w = self.patch_size

        if self.augment:
            self.aug_transform = build_crack_profile_augment(self.augment_profile)
        else:
            self.aug_transform = None


    def __len__(self):
        return len(self.images) * self.patches_per_image

    def _load_pair(self, img_name):
        if self._cache_size > 0:
            cached = self._cache.get(img_name)
            if cached is not None:
                self._cache.move_to_end(img_name)
                return cached

        record = self._record_by_name.get(img_name)
        if record is None:
            base_name = os.path.splitext(img_name)[0]
            raise FileNotFoundError(
                f"Mask not found for '{img_name}'. Expected '{base_name}{self.mask_prefix}.*' in {self.mask_dir}"
            )
        image = np.array(Image.open(record.image_path).convert("RGB"))
        mask = np.array(Image.open(record.mask_path).convert("L"))
        crop_metadata = build_crop_metadata(image, mask, crop_policy=self.crop_policy)
        if self._cache_size > 0:
            self._cache[img_name] = (image, mask, crop_metadata)
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return image, mask, crop_metadata

    # _random_patch removed in favor of Albumentations RandomCrop


    def __getitem__(self, idx):
        img_idx = int(idx) // self.patches_per_image
        img_idx = max(0, min(img_idx, len(self.images) - 1))
        img_name = self.images[img_idx]

        try:
            image_raw, mask_raw, crop_metadata = self._load_pair(img_name)
            image_np = ensure_three_channel_image(np.array(image_raw, copy=True))
            mask_np = normalize_binary_mask(np.array(mask_raw, copy=True))
            patch_h, patch_w = self.patch_size
            image_np, mask_np = pad_canvas_if_needed(
                image_np,
                mask_np,
                min_h=patch_h,
                min_w=patch_w,
                image_border_mode=0,
                image_fill=(0, 0, 0),
                label_border_mode=0,
                label_fill=0,
            )
            h, w = mask_np.shape[:2]
            y1, x1 = choose_smart_crack_crop_coords(
                image_np,
                mask_np,
                h,
                w,
                patch_h,
                patch_w,
                background_crop_prob=self.background_crop_prob,
                near_background_crop_prob=self.near_background_crop_prob,
                hard_negative_crop_prob=self.hard_negative_crop_prob,
                crop_policy=self.crop_policy,
                metadata=crop_metadata,
            )
            image_np, mask_np = crop_image_label(image_np, mask_np, y1=y1, x1=x1, th=patch_h, tw=patch_w)

            if self.aug_transform is not None:
                augmented = self.aug_transform(image=to_uint8_image(image_np), mask=(mask_np > 0).astype(np.uint8))
                image_np = augmented['image']
                mask_np = augmented['mask']

            # 3. To Tensor
            if self.image_transform is not None:
                image = self.image_transform(Image.fromarray(image_np))
            else:
                image = torch.from_numpy(image_np.astype(np.float32) / 255.0).permute(2, 0, 1)
                mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype).view(3, 1, 1)
                image = (image - mean) / std

            # Mask: keep binary labels robust across both float masks and uint8 0/1 masks.
            # torchvision.ToTensor() divides uint8 by 255, so passing a 0/1 uint8 mask would
            # turn positives into 1/255 and the later >0.5 threshold would erase them.
            mask_binary = ((np.asarray(mask_np) > 0).astype(np.uint8) * 255)
            if self.mask_transform is not None:
                mask = self.mask_transform(Image.fromarray(mask_binary, mode="L"))
                mask = (mask > 0.5).float()
            else:
                mask = (mask_binary > 0).astype(np.float32)
                mask = torch.from_numpy(mask).unsqueeze(0)

            return image, mask

        except Exception as e:
            if self.verbose:
                 print(f"RandomPatchDataset error at idx={idx} ({img_name}): {e}")
            return None
