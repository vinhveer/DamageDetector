import os
import random
import numpy as np
import torch
import albumentations as A
# from albumentations.pytorch import ToTensorV2  # Optional, but we can do manual cast
from PIL import Image
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .utils import _list_valid_images, build_mask_index, find_mask_path

class CrackDataset(Dataset):
    """
    High-Performance Crack Dataset Loader.
    
    Features:
    - Albumentations for Heavy Augmentation.
    - RAM Caching (256GB RAM Optimized).
    - Float32 Tensors.
    """
    def __init__(
        self,
        image_dir,
        mask_dir,
        transform=None,      # Legacy arg, ignored or mapped
        image_transform=None, # Legacy arg
        mask_transform=None,  # Legacy arg
        mask_prefix="auto",
        mask_index=None,
        augment=False,       # If True, use Heavy Augmentation
        patch_size=None,     # Ignored if output_size is set, we prefer resize
        patch_strategy="random",
        max_patch_tries=10,
        output_size=512,     # Force resize to this
        image_filenames=None,
        verbose=True,
        cache_data=False      # NEW: Persistent RAM Cache
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_prefix = str(mask_prefix)
        self.augment = augment
        self.output_size = int(output_size)
        self.verbose = verbose
        self.cache_data = cache_data

        # Check dirs
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

        if mask_index is None:
            mask_index = build_mask_index(mask_dir)
        self._mask_index = mask_index

        # Collect files
        if image_filenames is None:
            self.images = sorted(
                [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            )
        else:
            self.images = list(image_filenames)

        # Validate pairs
        valid_images = []
        for img in self.images:
            base_name = os.path.splitext(img)[0]
            mask_path = find_mask_path(mask_dir, base_name, mask_prefix=self.mask_prefix, mask_index=self._mask_index)
            if mask_path is not None:
                valid_images.append(img)
        self.images = valid_images
        
        if self.verbose:
            print(f"CrackDataset: Found {len(self.images)} pairs in {image_dir}")

        # --- RAM Caching ---
        self.cached_imgs = {}
        self.cached_masks = {}
        
        if self.cache_data:
            if self.verbose:
                print(f"CrackDataset: Pre-caching {len(self.images)} images into RAM...")
            
            def load_func(idx):
                name = self.images[idx]
                i_path = os.path.join(self.image_dir, name)
                base = os.path.splitext(name)[0]
                m_path = find_mask_path(self.mask_dir, base, self.mask_prefix, self._mask_index)
                
                # Load as uint8 numpy
                img_np = np.array(Image.open(i_path).convert("RGB"))
                mask_np = np.array(Image.open(m_path).convert("L"))
                return idx, img_np, mask_np

            # Use ThreadPool for fast IO
            with ThreadPoolExecutor(max_workers=16) as executor:
                results = list(tqdm(executor.map(load_func, range(len(self.images))), total=len(self.images), disable=not verbose))
            
            for idx, img, mask in results:
                self.cached_imgs[idx] = img
                self.cached_masks[idx] = mask
            
            if self.verbose:
                print("CrackDataset: Caching complete.")

        # --- Transforms Setup (Albumentations) ---
        # Base resize
        base_transforms = [
            A.Resize(height=self.output_size, width=self.output_size, interpolation=1) # Linear
        ]
        
        if self.augment:
            # Heavy Augmentation (Same as SAM)
            aug_transforms = [
                 A.HorizontalFlip(p=0.5),
                 A.VerticalFlip(p=0.5),
                 A.RandomRotate90(p=0.5),
                 # Geometry
                 A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=30, p=0.5),
                 A.OneOf([
                    A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
                 ], p=0.3),
                 # Color / Noise
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
                 # Occlusion/Shadow
                 A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.3),
                 A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8, fill_value=0, mask_fill_value=0, p=0.3),
            ]
            self.transform = A.Compose(aug_transforms + base_transforms)
        else:
            # Val/Test: Just Resize
            self.transform = A.Compose(base_transforms)

    @staticmethod
    def list_valid_images(image_dir, mask_dir, mask_prefix="auto", mask_index=None):
        return _list_valid_images(image_dir, mask_dir, mask_prefix=mask_prefix, mask_index=mask_index)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Load Data
        if self.cache_data and idx in self.cached_imgs:
            image = self.cached_imgs[idx] # uint8 HWC
            mask = self.cached_masks[idx] # uint8 HW
        else:
            img_name = self.images[idx]
            img_path = os.path.join(self.image_dir, img_name)
            base = os.path.splitext(img_name)[0]
            mask_path = find_mask_path(self.mask_dir, base, self.mask_prefix, self._mask_index)
            
            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"))

        # 2. Augment (Albumentations works on uint8 numpy)
        # mask needs to be passed significantly. Albumentations handles mask automatically.
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']

        # 3. Normalize & To Tensor
        # Image: [0, 255] uint8 -> [0, 1] float32 -> [C, H, W]
        image = (image.astype(np.float32) / 255.0)
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Mask: [0, 255] uint8 -> [0, 1] float32 -> [1, H, W]
        # Binarize first to be safe
        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask
