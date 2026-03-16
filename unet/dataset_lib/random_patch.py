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
        aug_prob=0.5,
        rotate_limit=30.0,
        brightness_limit=0.2,
        contrast_limit=0.2,
        negative_patch_prob=0.25,
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
        self.aug_prob = max(0.0, min(1.0, float(aug_prob)))
        self.rotate_limit = float(rotate_limit)
        self.brightness_limit = float(brightness_limit)
        self.contrast_limit = float(contrast_limit)
        self.negative_patch_prob = max(0.0, min(1.0, float(negative_patch_prob)))
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
        # 2. Augmentations (Only if augment=True)
        if self.augment:
            # Matches SAM GenericDataset
            self.aug_transform = A.Compose([
                # Safe Logic: Pad -> RandomCrop
                A.PadIfNeeded(min_height=patch_h, min_width=patch_w, border_mode=0, value=0, mask_value=0),
                A.RandomCrop(height=patch_h, width=patch_w, p=1.0),
                
                A.HorizontalFlip(p=self.aug_prob),
                A.VerticalFlip(p=min(1.0, self.aug_prob * 0.6)),
                A.RandomRotate90(p=min(1.0, self.aug_prob * 0.5)),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=0.0625,
                    rotate=(-self.rotate_limit, self.rotate_limit),
                    p=self.aug_prob,
                ),
                A.OneOf([
                    A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05), # Removed alpha_affine
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
                ], p=min(1.0, self.aug_prob * 0.6)),
                
                # Color/Noise Augmentations
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.RandomBrightnessContrast(
                        brightness_limit=self.brightness_limit,
                        contrast_limit=self.contrast_limit,
                    ),
                    A.RandomGamma(),            
                ], p=self.aug_prob),
                
                A.OneOf([
                    A.GaussNoise(),
                    A.MotionBlur(blur_limit=3),
                ], p=min(1.0, self.aug_prob * 0.5)),
                
                A.HueSaturationValue(p=min(1.0, self.aug_prob * 0.3)),
                
                # Environmental / Occlusion
                A.RandomShadow(
                    num_shadows_limit=(1, 3),
                    shadow_dimension=5,
                    shadow_roi=(0, 0.5, 1, 1),
                    p=min(1.0, self.aug_prob * 0.3),
                ),
                A.CoarseDropout(
                    num_holes_limit=(1, 8),
                    hole_height_range=(8, 32),
                    hole_width_range=(8, 32),
                    min_holes=None,
                    min_height=None,
                    min_width=None,
                    fill_value=0,
                    mask_fill_value=0,
                    p=min(1.0, self.aug_prob * 0.3),
                ),
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

            # 1. Random Crop with controllable positive/negative balance.
            best_sample = None
            best_positive = None
            best_negative = None
            keep_negative = random.random() < self.negative_patch_prob

            for _ in range(max(1, self.max_patch_tries)):
                cropped = self.crop_transform(image=image_np, mask=mask_np)
                has_crack = np.count_nonzero(cropped["mask"]) > 0
                if has_crack:
                    best_positive = cropped
                    if not keep_negative:
                        best_sample = cropped
                        break
                else:
                    best_negative = cropped

            if best_sample is None:
                if keep_negative and best_negative is not None:
                    best_sample = best_negative
                elif best_positive is not None:
                    best_sample = best_positive
                else:
                    best_sample = best_negative
            
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
            if self.image_transform is not None:
                image = self.image_transform(Image.fromarray(image_np))
            else:
                image = torch.from_numpy(image_np.astype(np.float32) / 255.0).permute(2, 0, 1)
                mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype).view(3, 1, 1)
                image = (image - mean) / std

            # Mask: [H, W] -> [1, H, W], float32 [0, 1]
            if self.mask_transform is not None:
                mask = self.mask_transform(Image.fromarray(mask_np))
                mask = (mask > 0.5).float()
            else:
                mask = (mask_np > 127).astype(np.float32)
                mask = torch.from_numpy(mask).unsqueeze(0)

            return image, mask

        except Exception as e:
            if self.verbose:
                 print(f"RandomPatchDataset error at idx={idx} ({img_name}): {e}")
            return None
