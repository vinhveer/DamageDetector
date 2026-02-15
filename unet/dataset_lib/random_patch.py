import os
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance
from collections import OrderedDict
from torch.utils.data import Dataset

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

        # --- Albumentations Setup ---
        patch_h, patch_w = self.patch_size
        
        # 1. Base crop (Random Crop)
        self.crop_transform = A.Compose([
            A.PadIfNeeded(min_height=patch_h, min_width=patch_w, border_mode=0, value=0, mask_value=0),
            A.RandomCrop(height=patch_h, width=patch_w)
        ])

        # 2. Augmentations (Only if augment=True)
        if self.augment:
            # Matches SAM GenericDataset
            self.aug_transform = A.Compose([
                # Safe Logic: Pad -> RandomCrop
                A.PadIfNeeded(min_height=patch_h, min_width=patch_w, border_mode=0, value=0, mask_value=0),
                A.RandomCrop(height=patch_h, width=patch_w, p=1.0),
                
                A.Affine(scale=(0.9, 1.1), translate_percent=0.0625, rotate=30, p=0.5),
                A.OneOf([
                    A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
                ], p=0.3),
                
                # Color/Noise Augmentations
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.RandomBrightnessContrast(),
                    A.RandomGamma(),            
                ], p=0.5),
                
                A.OneOf([
                    A.GaussNoise(),
                    A.MotionBlur(blur_limit=3),
                ], p=0.3),
                
                A.HueSaturationValue(p=0.3),
                
                # Environmental / Occlusion
                A.RandomShadow(num_shadows_limit=(1, 3), shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.3),
                A.CoarseDropout(num_holes_limit=8, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8, fill_value=0, mask_fill_value=0, p=0.3),
            ], is_check_shapes=False)
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

    # _random_patch removed in favor of Albumentations RandomCrop


    def __getitem__(self, idx):
        img_idx = int(idx) // self.patches_per_image
        img_idx = max(0, min(img_idx, len(self.images) - 1))
        img_name = self.images[img_idx]

        try:
            image_pil, mask_pil = self._load_pair(img_name)
            
            # Convert PIL to Numpy for Albumentations
            image_np = np.array(image_pil)
            mask_np = np.array(mask_pil)

            # 1. Random Crop (Try K times to find crack)
            best_sample = None
            
            for _ in range(max(1, self.max_patch_tries)):
                cropped = self.crop_transform(image=image_np, mask=mask_np)
                if np.count_nonzero(cropped['mask']) > 0:
                    best_sample = cropped
                    break
                best_sample = cropped # Fallback to last try
            
            if best_sample is None:
                return None

            image_np = best_sample['image']
            mask_np = best_sample['mask']

            # 2. Augment
            if self.aug_transform is not None:
                augmented = self.aug_transform(image=image_np, mask=mask_np)
                image_np = augmented['image']
                mask_np = augmented['mask']

            # 3. To Tensor
            # Image: [H, W, C] -> [C, H, W], float32 [0, 1]
            image = (image_np.astype(np.float32) / 255.0)
            image = torch.from_numpy(image).permute(2, 0, 1)

            # Mask: [H, W] -> [1, H, W], float32 [0, 1]
            mask = (mask_np > 127).astype(np.float32)
            mask = torch.from_numpy(mask).unsqueeze(0)

            return image, mask

        except Exception as e:
            if self.verbose:
                 print(f"RandomPatchDataset error at idx={idx} ({img_name}): {e}")
            return None

