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
        preprocess_mode="letterbox", # NEW: Control sizing strategy
        patches_per_image=1  # NEW: Number of crops per image per epoch
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_prefix = str(mask_prefix)
        self.augment = augment
        self.output_size = int(output_size)
        self.verbose = verbose
        self.cache_data = False 
        self.preprocess_mode = preprocess_mode # Store for __getitem__ logic

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
        
        # Normalization (ImageNet Standard)
        norm_transform = [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]

        if self.augment:
             # Heavy Augmentation w/o Sizing (Applied AFTER Smart Crop)
             aug_list = [
                 # Geometric (Safe)
                 A.HorizontalFlip(p=0.5),
                 A.VerticalFlip(p=0.5),
                 A.RandomRotate90(p=0.5),
                 A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.1), rotate=(-45, 45), p=0.5),
                 
                 # Distortions
                 A.OneOf([
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                    A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, p=1.0)
                 ], p=0.5),

                 # Occlusion
                 A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.3),
                 A.CoarseDropout(
                    max_holes=10, max_height=32, max_width=32, 
                    min_holes=1, min_height=8, min_width=8, 
                    fill_value=0, mask_fill_value=0, p=0.3
                 ),

                 # Weather Effects (Outdoor Robustness)
                 A.OneOf([
                     A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=1.0),
                     A.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=1.0),
                     A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, alpha_coef=0.08, p=1.0),
                 ], p=0.4),

                 # Color/Noise
                 A.OneOf([
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),            
                 ], p=0.5),
                 
                 A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.Blur(blur_limit=3, p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                 ], p=0.3),
                 
                 A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            ]
        else:
             aug_list = []

        # Construction Pipeline
        if preprocess_mode == "random_crop":
            # Smart Crop Logic handles sizing manually in __getitem__
            # self.transform only handles Augment + Normalize
            self.transform = A.Compose(aug_list + norm_transform, is_check_shapes=False)
            
        elif preprocess_mode == "letterbox":
            size_transforms = [
                 A.LongestMaxSize(max_size=self.output_size, interpolation=cv2.INTER_LINEAR),
                 A.PadIfNeeded(
                     min_height=self.output_size, min_width=self.output_size, 
                     border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0
                 )
            ]
            self.transform = A.Compose(size_transforms + aug_list + norm_transform, is_check_shapes=False)
            
        elif preprocess_mode == "resize":
            size_transforms = [
                 A.Resize(height=self.output_size, width=self.output_size, interpolation=cv2.INTER_LINEAR)
            ]
            self.transform = A.Compose(size_transforms + aug_list + norm_transform, is_check_shapes=False)
            
        else:
             # Legacy/Default: Pad -> Crop
             # Or Fallback to Resize if unknown
             if self.augment:
                  # Explicitly insert Pad/Crop logic to aug list for legacy
                  legacy_aug = list(aug_list)
                  legacy_aug.insert(0, A.PadIfNeeded(min_height=self.output_size, min_width=self.output_size, border_mode=0, value=0, mask_value=0))
                  legacy_aug.insert(1, A.RandomCrop(height=self.output_size, width=self.output_size))
                  self.transform = A.Compose(legacy_aug + norm_transform, is_check_shapes=False)
             else:
                  # Just Resize for Val Fallback
                  self.transform = A.Compose([A.Resize(self.output_size, self.output_size)] + norm_transform, is_check_shapes=False)

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
        
        image = cv2.imread(img_path)
        if image is None: raise FileNotFoundError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: raise FileNotFoundError(f"Failed to load mask: {mask_path}")
        
        if image.shape[:2] != mask.shape[:2]:
            h, w = image.shape[:2]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 2. Smart Crop (if enabled)
        if self.preprocess_mode == "random_crop":
            # Pad if needed first
            h, w = image.shape[:2]
            pad_h = max(0, self.output_size - h)
            pad_w = max(0, self.output_size - w)
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
                mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                h, w = image.shape[:2] # Update sizes

            # Smart Crop Logic
            new_h, new_w = self.output_size, self.output_size
            
            # Decide: Positive Crop or Random Crop?
            # If augment=True, 80% chance to pick positive.
            # If augment=False (Val), we still want to see cracks? 
            # Ideally Validation should use sliding window, but for 'training loop validation', 
            # let's just center crop or random crop. 
            # Let's simple apply RandomCrop for Val to be consistent with train loader or just Resize.
            # Actually, Val with RandomCrop is unstable metrics. 
            # But we stick to what requested: "random_crop" in config applies to both.
            
            do_smart = False
            # Smart Crop: Always 100% (Ensure every sample contains cracks)
            # This aligns with SAM Fine-tune strategy.
            threshold = 1.0
            if random.random() < threshold:
                # Find positive pixels
                y_inds, x_inds = np.where(mask > 0)
                if len(y_inds) > 0:
                    do_smart = True
                    if self.augment:
                        # Train: Pick a random crack pixel
                        idx = random.randint(0, len(y_inds) - 1)
                        cy, cx = y_inds[idx], x_inds[idx]
                    else:
                        # Val: Pick deterministic center (median) of crack
                        cy = int(np.median(y_inds))
                        cx = int(np.median(x_inds))
                    
                    # Random jitter around center?
                    start_y1 = max(0, cy - new_h // 2)
                    y1 = min(start_y1, h - new_h)
                    y1 = int(max(0, y1))

                    if self.augment:
                        # Allow random jitter around center
                        min_y1 = max(0, cy - new_h + 1)
                        max_y1 = min(h - new_h, cy)
                        max_y1 = max(max_y1, min_y1)
                        y1 = random.randint(min_y1, max_y1)

                    start_x1 = max(0, cx - new_w // 2)
                    x1 = min(start_x1, w - new_w)
                    x1 = int(max(0, x1))

                    if self.augment:
                        min_x1 = max(0, cx - new_w + 1)
                        max_x1 = min(w - new_w, cx)
                        max_x1 = max(max_x1, min_x1)
                        x1 = random.randint(min_x1, max_x1)
                    
                    image = image[y1:y1+new_h, x1:x1+new_w]
                    mask = mask[y1:y1+new_h, x1:x1+new_w]
            
            if not do_smart:
                # Fallback to Random Crop (Pure Random)
                if self.augment:
                    if h > new_h:
                        y1 = random.randint(0, h - new_h)
                    else: y1 = 0
                    
                    if w > new_w:
                        x1 = random.randint(0, w - new_w)
                    else: x1 = 0
                else:
                    # Valid/Test: Center Crop Fallback
                    y1 = max(0, (h - new_h) // 2)
                    x1 = max(0, (w - new_w) // 2)

                image = image[y1:y1+new_h, x1:x1+new_w]
                mask = mask[y1:y1+new_h, x1:x1+new_w]
            else:
                 # Smart Crop logic determined above
                 pass # Already handled inside the if do_smart block? 
                 # Wait, I need to edit the INSIDE of do_smart block too. 
                 # The previous tool call view_file shows do_smart block ends at 313.
                 # I need to target the whole block.


        # 3. Augment & Normalize
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']

        # 4. To Tensor
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask

