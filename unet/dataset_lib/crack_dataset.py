import os
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from .utils import _list_valid_images, _normalize_patch_size, _pad_to_min_size, build_mask_index, find_mask_path


class CrackDataset(Dataset):
    """
    Crack dataset loader.

    Responsibilities:
    - Loads images and their corresponding masks.
    - Optionally applies data augmentation to improve generalization.
    - Returns tensors in a consistent, model-ready format.

    Args:
        image_dir: Directory containing input images.
        mask_dir: Directory containing mask images.
        transform: Optional transform applied to both image and mask (legacy).
        image_transform: Transform applied only to the image.
        mask_transform: Transform applied only to the mask.
        augment: Whether to enable random data augmentation.
        patch_size: If set, return a cropped patch (e.g., 256) from the original image.
        patch_strategy: "random" (optionally biased to crack pixels) or "center".
        max_patch_tries: Number of random tries to find a patch containing cracks.
        output_size: Used for safe fallback tensors if a sample fails to load.
        image_filenames: Optional list of image filenames to use (pre-split train/val).
        verbose: Print dataset summary.
    """
    def __init__(
        self,
        image_dir,
        mask_dir,
        transform=None,
        image_transform=None,
        mask_transform=None,
        mask_prefix="auto",
        mask_index=None,
        augment=False,
        patch_size=None,
        patch_strategy="random",
        max_patch_tries=10,
        output_size=256,
        image_filenames=None,
        verbose=True,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.mask_prefix = str(mask_prefix)
        self.augment = augment
        self.patch_size = patch_size
        self.patch_strategy = patch_strategy
        self.max_patch_tries = max_patch_tries
        self.output_size = int(output_size)
        self.verbose = verbose

        if self.image_transform is None and self.mask_transform is None and self.transform is not None:
            # Backwards compatibility: previous code passed a single transform for both.
            self.image_transform = self.transform
            self.mask_transform = self.transform

        # Check if directories exist
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

        if mask_index is None:
            mask_index = build_mask_index(mask_dir)
        self._mask_index = mask_index

        self.patch_size = self._normalize_patch_size(self.patch_size)

        # Collect image filenames (only image extensions), or use the provided list.
        if image_filenames is None:
            self.images = sorted(
                [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            )
        else:
            self.images = list(image_filenames)

        # Validate pairs: keep only images with a matching mask.
        valid_images = []
        for img in self.images:
            base_name = os.path.splitext(img)[0]
            mask_path = find_mask_path(
                mask_dir,
                base_name,
                mask_prefix=self.mask_prefix,
                mask_index=self._mask_index,
            )
            if mask_path is not None:
                valid_images.append(img)

        self.images = valid_images
        if self.verbose:
            print(f"Found {len(self.images)} valid image-mask pairs")

        # Print the first 2 samples to help debugging.
        if self.verbose:
            for i in range(min(2, len(self.images))):
                img_name = self.images[i]
                base_name = os.path.splitext(img_name)[0]
                mask_path = find_mask_path(
                    mask_dir,
                    base_name,
                    mask_prefix=self.mask_prefix,
                    mask_index=self._mask_index,
                )
                print(f"Image {i}: {img_name}")
                print(f"Mask  {i}: {os.path.basename(mask_path) if mask_path else '(missing)'}")

    @staticmethod
    def list_valid_images(image_dir, mask_dir, mask_prefix="auto", mask_index=None):
        if mask_index is None:
            mask_index = build_mask_index(mask_dir)
        return _list_valid_images(
            image_dir, mask_dir, mask_prefix=mask_prefix, mask_index=mask_index
        )

    @staticmethod
    def _normalize_patch_size(patch_size):
        return _normalize_patch_size(patch_size)

    @staticmethod
    def _pad_to_min_size(img: Image.Image, min_w: int, min_h: int, fill):
        return _pad_to_min_size(img, min_w, min_h, fill)

    def _crop_patch(self, image: Image.Image, mask: Image.Image):
        patch_w, patch_h = self.patch_size

        image = self._pad_to_min_size(image, patch_w, patch_h, fill=(0, 0, 0))
        mask = self._pad_to_min_size(mask, patch_w, patch_h, fill=0)

        w, h = image.size
        if self.patch_strategy == "center":
            left = max(0, (w - patch_w) // 2)
            top = max(0, (h - patch_h) // 2)
            box = (left, top, left + patch_w, top + patch_h)
            return image.crop(box), mask.crop(box)

        # Random crop, optionally biased toward patches that contain cracks.
        best = None
        for _ in range(max(1, int(self.max_patch_tries))):
            left = 0 if w == patch_w else random.randint(0, w - patch_w)
            top = 0 if h == patch_h else random.randint(0, h - patch_h)
            box = (left, top, left + patch_w, top + patch_h)
            img_patch = image.crop(box)
            mask_patch = mask.crop(box)
            best = (img_patch, mask_patch)

            # Fast check: if this patch has any non-zero pixels, it likely contains crack.
            if mask_patch.getbbox() is not None:
                return img_patch, mask_patch

        return best

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Return one sample (image, mask), optionally applying augmentation.

        Notes:
        - Image and mask must receive the same geometric transforms to stay aligned.
        - Brightness/contrast should only be applied to the image (not the mask).
        - Exceptions are handled so a single bad sample doesn't stop training.
        """
        try:
            # Load image
            img_name = self.images[idx]
            img_path = os.path.join(self.image_dir, img_name)

            # Build mask path using the unified naming rule: {base}{mask_prefix}.*
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

            # Read images and masks, and convert to appropriate formats.
            image = Image.open(img_path).convert('RGB')  # Ensure 3 channels
            mask = Image.open(mask_path).convert('L')  # Single-channel grayscale

            # Data augmentation (randomly applied during training)
            if self.augment:
                if random.random() < 0.5:
                    rot = random.choice([Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
                    image = image.transpose(rot)
                    mask = mask.transpose(rot)

                if random.random() < 0.5:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

                if random.random() < 0.3:
                    enhancer = ImageEnhance.Brightness(image)
                    image = enhancer.enhance(random.uniform(0.8, 1.2))

                if random.random() < 0.3:
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(random.uniform(0.8, 1.2))

            # Patch-based training: crop a 256x256 (or configured) patch at original resolution.
            if self.patch_size is not None:
                image, mask = self._crop_patch(image, mask)

            # Apply additional transforms (resize, convert to tensor, etc.)
            if self.image_transform is not None:
                image = self.image_transform(image)
            else:
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
            else:
                mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0

            # Ensure the mask is binary (0 or 1).
            mask = (mask > 0.5).float()  # Pixels > 0.5 are cracks (1), else background (0)

            return image, mask

        except Exception as e:
            # Error handling: prevent a single sample error from stopping the entire training.
            print(f"Error processing image at index {idx}: {e}")
            # Skip bad samples (collate_fn should drop None).
            return None
