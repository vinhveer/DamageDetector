import os
import random
import numpy as np
import torch
import cv2
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
        cache_data=False,     # NEW: Persistent RAM Cache
        preprocess_mode="letterbox" # NEW: Control sizing strategy
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_prefix = str(mask_prefix)
        self.augment = augment
        self.output_size = int(output_size)
        self.verbose = verbose
        # Force cache_data to False to prevent RAM issues on Windows with multiprocessing
        self.cache_data = False 
        if cache_data and self.verbose:
             print("Warning: cache_data disabled by default to prevent stability issues.")

        # Check dirs
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

        # Validate pairs and Pre-resolve paths
        self.images = []
        self.mask_paths = []
        
        # Temp local index for init only
        if mask_index is None:
             mask_index = build_mask_index(mask_dir)

        # Collect candidate files
        if image_filenames is None:
            # List directory if no filenames provided
            valid_candidate_images = sorted(
                [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            )
        else:
            valid_candidate_images = list(image_filenames)
             
        for img in valid_candidate_images: # We need to iterate over the candidate list passed or found
            base_name = os.path.splitext(img)[0]
            mask_path = find_mask_path(mask_dir, base_name, self.mask_prefix, mask_index=mask_index)
            if mask_path is not None:
                self.images.append(img)
                self.mask_paths.append(mask_path)
        
        # Clean up to prevent pickling large dicts to workers
        del mask_index
        self._mask_index = None 

        if self.verbose:
            print(f"CrackDataset: Found {len(self.images)} pairs in {image_dir}")
            print(f"CrackDataset: Preprocess Mode: {preprocess_mode} | Augment: {self.augment}")

        # --- RAM Caching ---
        self.cached_imgs = {}
        self.cached_masks = {}
        
        if self.cache_data:
            if self.verbose:
                print(f"CrackDataset: Pre-caching {len(self.images)} images into RAM...")
            
            def load_func(idx):
                name = self.images[idx]
                i_path = os.path.join(self.image_dir, name)
                m_path = self.mask_paths[idx] # Direct access
                
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

                return idx, img_cv, mask_cv

            # Use ThreadPool for fast IO
            with ThreadPoolExecutor(max_workers=16) as executor:
                results = list(tqdm(executor.map(load_func, range(len(self.images))), total=len(self.images), disable=not verbose))
            
            for idx, img, mask in results:
                self.cached_imgs[idx] = img
                self.cached_masks[idx] = mask
            
            if self.verbose:
                print("CrackDataset: Caching complete.")

        # --- Transforms Setup (Albumentations) ---
        
        # Define Sizing Strategy
        size_transforms = []
        if preprocess_mode == "letterbox":
            # Letterbox: Resize longest side to target, pad short side. Preserves Aspect Ratio.
            size_transforms = [
                A.LongestMaxSize(max_size=self.output_size, interpolation=cv2.INTER_LINEAR),
                A.PadIfNeeded(
                    min_height=self.output_size, 
                    min_width=self.output_size, 
                    border_mode=cv2.BORDER_CONSTANT, 
                    pad_cval=0, 
                    pad_cval_mask=0
                )
            ]
        elif preprocess_mode == "resize":
            # Resize: Squash to target.
            size_transforms = [
                 A.Resize(height=self.output_size, width=self.output_size, interpolation=cv2.INTER_LINEAR)
            ]
        elif preprocess_mode == "patch":
             # Handled externally usually, but if here, do nothing or resize?
             # Assuming patch mode handles its own slicing, but CrackDataset expects 1-1 image.
             # Fallback to resize.
             size_transforms = [
                 A.Resize(height=self.output_size, width=self.output_size, interpolation=cv2.INTER_LINEAR)
            ]
        else:
            passing = True

        # Normalization (ImageNet Standard)
        norm_transform = [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]

        if self.augment:
            # Heavy Augmentation
            aug_list = [
                 # Geometric (Safe)
                 A.HorizontalFlip(p=0.5),
                 A.VerticalFlip(p=0.5),
                 A.RandomRotate90(p=0.5),
                 # ShiftScaleRotate is deprecated in favor of Affine
                 A.Affine(scale=(0.9, 1.1), translate_percent=(0.0625, 0.0625), rotate=(-30, 30), p=0.5),
                 
                 # Distortions
                 A.OneOf([
                    A.GridDistortion(p=1.0),
                    # alpha_affine is deprecated
                    A.ElasticTransform(alpha=1, sigma=50, p=1.0)
                 ], p=0.3),

                 # Color/Noise
                 A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.RandomBrightnessContrast(),
                    A.RandomGamma(),            
                 ], p=0.5),
                 
                 A.OneOf([
                    A.GaussNoise(),
                    A.Blur(blur_limit=3),
                 ], p=0.3),
                 
                 A.HueSaturationValue(p=0.3),
            ]
            
            # Special case for "random_crop" mode legacy
            if preprocess_mode not in ["letterbox", "resize"]:
                 # Legacy Random Crop logic: Pad -> Crop
                 aug_list.insert(0, A.PadIfNeeded(min_height=self.output_size, min_width=self.output_size, border_mode=0, pad_cval=0, pad_cval_mask=0))
                 aug_list.insert(1, A.RandomCrop(height=self.output_size, width=self.output_size))
                 # No size_transforms needed at end if we cropped
                 self.transform = A.Compose(aug_list + norm_transform, is_check_shapes=False)
            else:
                 # Standard: Augment -> Resize/Letterbox -> Normalize
                 full_list = size_transforms + aug_list + norm_transform
                 self.transform = A.Compose(full_list, is_check_shapes=False)

        else:
            # Val/Test: Just Size -> Normalize
            self.transform = A.Compose(size_transforms + norm_transform, is_check_shapes=False)

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
            mask_path = self.mask_paths[idx] # Direct access
            
            # Load using OpenCV (Robust)
            image = cv2.imread(img_path)
            if image is None:
                 raise FileNotFoundError(f"Failed to load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                 raise FileNotFoundError(f"Failed to load mask: {mask_path}")
            
            # Check Size Consistency (Critical Step!)
            if image.shape[:2] != mask.shape[:2]:
                h, w = image.shape[:2]
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 2. Augment (Albumentations works on uint8 numpy)
        # mask needs to be passed significantly. Albumentations handles mask automatically.
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']

        # 3. To Tensor (Normalize done in transform)
        # Image: [H, W, C] float32 -> [C, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Mask: [0, 255] uint8 -> [0, 1] float32 -> [1, H, W]
        # Binarize first to be safe
        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask
