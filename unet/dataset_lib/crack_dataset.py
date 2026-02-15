import os
import random
import numpy as np
import torch
import cv2
import albumentations as A
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
                 
                 # Distortions (Reduced: Too heavy deformation breaks crack continuity)
                 # A.OneOf([
                 #    A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                 #    A.OpticalDistortion(distort_limit=1, p=1.0),
                 #    A.ElasticTransform(alpha=1, sigma=50, p=1.0)
                 # ], p=0.1),

                 # Occlusion (Disabled: CoarseDropout creates holes that look like disconnected cracks)
                 # A.RandomShadow(num_shadows_limit=(1, 3), shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.1),
                 # A.CoarseDropout(
                 #    num_holes_range=(1, 10), hole_height_range=(8, 32), hole_width_range=(8, 32), 
                 #    p=0.1
                 # ),

                 # Weather Effects (Disabled: Rain/Snow adds artifacts confusing with cracks)
                 # A.OneOf([
                 #     A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=1.0),
                 #     A.RandomSnow(brightness_coeff=2.5, snow_point_range=(0.3, 0.5), p=1.0),
                 #     A.RandomFog(fog_coef_range=(0.3, 0.5), alpha_coef=0.08, p=1.0),
                 # ], p=0.1),

                 # Color/Noise (Kept but reduced intensity)
                 A.OneOf([
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),            
                 ], p=0.5),
                 
                 A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
                    A.Blur(blur_limit=3, p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                 ], p=0.2),
                 
                 # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            ]
        else:
             aug_list = []

        # Construction Pipeline (Heavy Augmentation WITHOUT CROP/RESIZE)
        # Sizing is handled manually in __getitem__ via Smart Crop
        if self.augment:
             self.transform = A.Compose([
                 A.HorizontalFlip(p=0.5),
                 A.VerticalFlip(p=0.5),
                 A.RandomRotate90(p=0.5),
                 
                 # Geometric (Affine)
                 A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.1), rotate=(-45, 45), p=0.5),
                 
                 # Color/Noise (Moderate)
                 A.OneOf([
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),            
                 ], p=0.5),
                 
                 A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
                    A.Blur(blur_limit=3, p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                 ], p=0.2),
                 
                 # HueSaturationValue (Disabled)
                 # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            ], is_check_shapes=False)
        else:
             self.transform = A.Compose([], is_check_shapes=False)

        # Normalization (ImageNet - Specific to UNet Backbones)
        self.normalize = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0)
        ])

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
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
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
            # Default: Smart Crop (Random Crop on Train, Center/Median on Val)
            # Find cracks (Optimized: Resize mask to 1/8 to find coords quickly)
            small_scale = 0.125 # 1/8
            mask_small = cv2.resize(mask, (0, 0), fx=small_scale, fy=small_scale, interpolation=cv2.INTER_NEAREST)
            y_inds_small, x_inds_small = np.where(mask_small > 0)
            
            # Map back to original scale
            if len(y_inds_small) > 0:
                # Approximate coords
                y_inds = (y_inds_small / small_scale).astype(int)
                x_inds = (x_inds_small / small_scale).astype(int)
            else:
                y_inds, x_inds = [], []
            
            th, tw = target_h, target_w
            
            # Pad if smaller than crop size
            if w < tw or h < th:
                pad_w = max(0, tw - w)
                pad_h = max(0, th - h)
                image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
                mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                h, w = image.shape[:2]

            if len(y_inds) > 0:
                # 30% chance to pick random crop (learn background), 70% force crack
                if self.augment and random.random() < 0.3:
                     y1 = random.randint(0, max(0, h - th))
                     x1 = random.randint(0, max(0, w - tw))
                elif self.augment:
                    # Train: Smart Random (Force Crack)
                    idx = random.randint(0, len(y_inds) - 1)
                    cy, cx = y_inds[idx], x_inds[idx]
                    y1_min = max(0, cy - th + 1)
                    x1_min = max(0, cx - tw + 1)
                    y1_max = min(h - th, cy)
                    x1_max = min(w - tw, cx)

                    # Clamp ranges to valid crop space
                    max_y1 = max(0, h - th)
                    max_x1 = max(0, w - tw)
                    y1_min = max(0, min(y1_min, max_y1))
                    x1_min = max(0, min(x1_min, max_x1))
                    y1_max = max(0, min(y1_max, max_y1))
                    x1_max = max(0, min(x1_max, max_x1))
                    if y1_max < y1_min:
                        y1_max = y1_min
                    if x1_max < x1_min:
                        x1_max = x1_min

                    y1 = random.randint(y1_min, y1_max)
                    x1 = random.randint(x1_min, x1_max)
                else:
                    # Val: Median Center
                    cy, cx = int(np.median(y_inds)), int(np.median(x_inds))
                    y1 = max(0, min(cy - th // 2, h - th))
                    x1 = max(0, min(cx - tw // 2, w - tw))
            else:
                # No crack -> Random (Train) or Center (Val)
                if self.augment:
                    y1 = random.randint(0, max(0, h - th))
                    x1 = random.randint(0, max(0, w - tw))
                else:
                    y1 = max(0, (h - th) // 2)
                    x1 = max(0, (w - tw) // 2)

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
        final_image = np.zeros((target_h, target_w, 3), dtype=np.float32)
        
        h_aug, w_aug = image_aug.shape[:2]
        
        # Safety Resize (in case aug changed size)
        if h_aug != target_h or w_aug != target_w:
             image_aug = cv2.resize(image_aug, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
             mask_aug = cv2.resize(mask_aug, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
             h_aug, w_aug = target_h, target_w

        # Paste (should be full cover usually)
        pad_top = max(0, (target_h - h_aug) // 2)
        pad_left = max(0, (target_w - w_aug) // 2)
        
        if len(image_aug.shape) == 2:
             final_image[pad_top:pad_top+h_aug, pad_left:pad_left+w_aug, 0] = image_aug
             final_image[pad_top:pad_top+h_aug, pad_left:pad_left+w_aug, 1] = image_aug
             final_image[pad_top:pad_top+h_aug, pad_left:pad_left+w_aug, 2] = image_aug
        elif image_aug.shape[2] == 1:
             ck = image_aug[:, :, 0]
             for k in range(3): final_image[pad_top:pad_top+h_aug, pad_left:pad_left+w_aug, k] = ck
        else:
             final_image[pad_top:pad_top+h_aug, pad_left:pad_left+w_aug, :] = image_aug
             
        # Mask Binarization
        final_mask = np.zeros((target_h, target_w), dtype=np.float32)
        # Handle mask resize/pad
        final_mask[pad_top:pad_top+h_aug, pad_left:pad_left+w_aug] = mask_aug
        
        # Thresholding: 0-255 -> 0.0/1.0
        # Use 127 as safe threshold for JPEG artifacts or soft edges
        final_mask = (final_mask > 127).astype(np.float32)

        image = torch.from_numpy(final_image).permute(2, 0, 1).float() 
        mask = torch.from_numpy(final_mask).float().unsqueeze(0)       
        
        return image, mask



