import os
import random
import numpy as np
from torch_runtime import torch
import cv2
import albumentations as A
from PIL import Image
from torch_runtime import Dataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from ..core import (
    build_crop_metadata,
    build_crack_profile_augment,
    choose_centered_foreground_crop_coords,
    choose_smart_crack_crop_coords,
    list_valid_pairs,
    pad_canvas_if_needed,
    resize_with_center_padding,
)
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
        cache_data=False,     # NEW: Persistent RAM Cache
        preprocess_mode="letterbox", # NEW: Control sizing strategy
        patches_per_image=1,  # NEW: Number of crops per image per epoch
        aug_prob=0.5,
        rotate_limit=10.0,
        brightness_limit=0.2,
        contrast_limit=0.2,
        augment_profile="balanced",
        crop_policy="smart",
        background_crop_prob=0.2,
        near_background_crop_prob=0.15,
        hard_negative_crop_prob=0.1,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_prefix = str(mask_prefix)
        self.augment = augment
        self.output_size = int(output_size)
        self.verbose = verbose
        self.cache_data = bool(cache_data)
        self.preprocess_mode = preprocess_mode # Store for __getitem__ logic
        self.aug_prob = max(0.0, min(1.0, float(aug_prob)))
        self.rotate_limit = float(rotate_limit)
        self.brightness_limit = float(brightness_limit)
        self.contrast_limit = float(contrast_limit)
        self.augment_profile = str(augment_profile or "balanced").strip().lower()
        self.crop_policy = str(crop_policy or "smart").strip().lower()
        self.background_crop_prob = float(background_crop_prob)
        self.near_background_crop_prob = float(near_background_crop_prob)
        self.hard_negative_crop_prob = float(hard_negative_crop_prob)

        if cache_data and self.verbose:
             print("CrackDataset: cache_data enabled. This may increase RAM usage significantly.")

        # Check dirs
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

        if mask_index is None:
            mask_index = build_mask_index(mask_dir)
        records = list_valid_pairs(
            image_dir,
            mask_dir,
            mask_prefix=self.mask_prefix,
            image_filenames=image_filenames,
            mask_index=mask_index,
        )
        self.images = [record.image_name for record in records]
        self.mask_paths = [record.mask_path for record in records]
        
        # Multi-Patch Training (Replication)
        if patches_per_image > 1:
            if self.verbose:
                print(f"CrackDataset: Multiplying dataset by {patches_per_image}x (Random Crops)")
            self.images = self.images * patches_per_image
            self.mask_paths = self.mask_paths * patches_per_image
        
        # Clean up to prevent pickling large dicts to workers
        del mask_index
        self._mask_index = None 

        if self.verbose:
            print(f"CrackDataset: Found {len(self.images)} pairs in {image_dir}")
            print(f"CrackDataset: Preprocess Mode: {preprocess_mode} | Augment: {self.augment}")

        # --- RAM Caching ---
        self.cached_imgs = {}
        self.cached_masks = {}
        self.cached_meta = {}
        
        if self.cache_data:
            if self.verbose:
                print(f"CrackDataset: Pre-caching {len(self.images)} images into RAM...")
            
            unique_items = []
            seen_items = set()
            for name, m_path in zip(self.images, self.mask_paths):
                key = (name, m_path)
                if key in seen_items:
                    continue
                seen_items.add(key)
                unique_items.append(key)

            def load_func(item):
                name, m_path = item
                i_path = os.path.join(self.image_dir, name)
                
                # Load as uint8 numpy via OpenCV (Standardize EXIF/Rotation handling)
                # Load Image
                img_cv = cv2.imread(i_path)
                if img_cv is None:
                    # Fallback or error? Let's just create a black image to avoid crash in thread
                    print(f"Warning: Failed to load image {name}")
                    img_cv = np.zeros((512, 512, 3), dtype=np.uint8)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

                # Load Mask (Grayscale)
                mask_cv = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
                if mask_cv is None:
                    print(f"Warning: Failed to load mask {name} (path: {m_path})")
                    mask_cv = np.zeros(img_cv.shape[:2], dtype=np.uint8)
                
                # Check Size Consistency (Critical Step to fix Aspect Ratio Mismatch!)
                if img_cv.shape[:2] != mask_cv.shape[:2]:
                    # Force Resize mask to match image
                    h, w = img_cv.shape[:2]
                    mask_cv = cv2.resize(mask_cv, (w, h), interpolation=cv2.INTER_NEAREST)

                crop_metadata = build_crop_metadata(img_cv, mask_cv, crop_policy=self.crop_policy)
                return name, m_path, img_cv, mask_cv, crop_metadata

            # Use ThreadPool for fast IO
            with ThreadPoolExecutor(max_workers=16) as executor:
                results = list(tqdm(executor.map(load_func, unique_items), total=len(unique_items), disable=not verbose))
            
            for name, m_path, img, mask, crop_metadata in results:
                cache_key = f"{name}|{m_path}"
                self.cached_imgs[cache_key] = img
                self.cached_masks[cache_key] = mask
                self.cached_meta[cache_key] = crop_metadata
            
            if self.verbose:
                print("CrackDataset: Caching complete.")

        if self.augment:
             self.transform = build_crack_profile_augment(self.augment_profile)
        else:
             self.transform = A.Compose([], is_check_shapes=False)

        self.normalize = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0)])

    @staticmethod
    def list_valid_images(image_dir, mask_dir, mask_prefix="auto", mask_index=None):
        return _list_valid_images(image_dir, mask_dir, mask_prefix=mask_prefix, mask_index=mask_index)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Load Data
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = self.mask_paths[idx] 
        cache_key = f"{img_name}|{mask_path}"

        if self.cache_data and cache_key in self.cached_imgs:
            image = np.array(self.cached_imgs[cache_key], copy=True)
            mask = np.array(self.cached_masks[cache_key], copy=True)
            crop_metadata = self.cached_meta.get(cache_key)
        else:
            image = cv2.imread(img_path)
            if image is None: raise FileNotFoundError(f"Failed to load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: raise FileNotFoundError(f"Failed to load mask: {mask_path}")
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            crop_metadata = build_crop_metadata(image, mask, crop_policy=self.crop_policy)
        
        # --- Preprocessing Strategy Switch ---
        target_h, target_w = self.output_size, self.output_size
        h, w = image.shape[:2]

        if self.preprocess_mode == "letterbox":
            # Letterbox Resize (Keep Aspect Ratio + Pad)
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # Create buffers
            image_final_buffer = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            mask_final_buffer = np.zeros((target_h, target_w), dtype=np.uint8)
            
            # Paste Center
            pad_top = (target_h - new_h) // 2
            pad_left = (target_w - new_w) // 2
            
            image_final_buffer[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = image_resized
            mask_final_buffer[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = mask_resized
            
            image = image_final_buffer
            mask = mask_final_buffer

        elif self.preprocess_mode == "resize":
            # Simple Stretch
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        else: 
            th, tw = target_h, target_w
            image, mask = pad_canvas_if_needed(
                image,
                mask,
                min_h=th,
                min_w=tw,
                image_border_mode=cv2.BORDER_CONSTANT,
                image_fill=(0, 0, 0),
                label_border_mode=cv2.BORDER_CONSTANT,
                label_fill=0,
            )
            h, w = image.shape[:2]
            if self.augment:
                y1, x1 = choose_smart_crack_crop_coords(
                    image,
                    mask,
                    h,
                    w,
                    th,
                    tw,
                    background_crop_prob=self.background_crop_prob,
                    near_background_crop_prob=self.near_background_crop_prob,
                    hard_negative_crop_prob=self.hard_negative_crop_prob,
                    crop_policy=self.crop_policy,
                    metadata=crop_metadata,
                )
            else:
                y1, x1 = choose_centered_foreground_crop_coords(mask, h, w, th, tw, metadata=crop_metadata)

            image = image[y1:y1+th, x1:x1+tw]
            mask = mask[y1:y1+th, x1:x1+tw]

        # 2. Augment
        # Image/Mask are now (H, W, 3) and (H, W) uint8
        if self.augment:
            augmented = self.transform(image=image, mask=mask)
            image_aug = augmented['image']
            mask_aug = augmented['mask']
        else:
            image_aug = image
            mask_aug = mask
        
        # 3. Normalize (Standardize: (X - Mean) / Std)
        # Returns float32
        if self.normalize:
            normed = self.normalize(image=image_aug)
            image_aug = normed['image']
        else:
            image_aug = image_aug.astype(np.float32) / 255.0 # Fallback if no normalize

        # 4. To Tensor
        # Robust Channel Handling
        target_h, target_w = self.output_size, self.output_size
        if image_aug.shape[:2] != (target_h, target_w):
            image_aug = cv2.resize(image_aug, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            mask_aug = cv2.resize(mask_aug, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        final_image, final_mask = resize_with_center_padding(
            image_aug,
            mask_aug.astype(np.float32),
            target_h=target_h,
            target_w=target_w,
            normalize_image=False,
        )
        if final_image.shape[2] == 1:
            final_image = np.repeat(final_image, 3, axis=2)
        final_mask = (final_mask > 0).astype(np.float32)

        image = torch.from_numpy(final_image).permute(2, 0, 1).float() 
        mask = torch.from_numpy(final_mask).float().unsqueeze(0)       
        
        return image, mask
