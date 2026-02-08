import os
import random
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import glob
import cv2
import albumentations as A

def random_rot_flip(image, label):
    # Deprecated: Handled by Albumentations
    pass

def random_rotate(image, label):
    # Deprecated: Handled by Albumentations
    pass

class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res
        
        # Define Albumentations pipeline (Heavy Augmentation WITHOUT CROP)
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            
            # Geometric Augmentations (Affine) - Aligned with UNet
            A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.1), rotate=(-45, 45), p=0.5),
            
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, p=1.0)
            ], p=0.5),
            
            # Color/Noise Augmentations
            A.OneOf([
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),            
            ], p=0.5),
            
            # Weather Effects (Outdoor Robustness)
            A.OneOf([
                A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=1.0),
                A.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=1.0),
                A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, alpha_coef=0.08, p=1.0),
            ], p=0.4),

            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.Blur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            
            # Environmental / Occlusion
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.3),
            A.CoarseDropout(max_holes=10, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8, fill_value=0, mask_fill_value=0, p=0.3),
        ], is_check_shapes=False)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # Ensure numpy
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if isinstance(label, torch.Tensor):
            label = label.numpy()
            
        # --- Smart Crop Implementation ---
        # Ensure image has 3 dimensions (H, W, C)
        if len(image.shape) == 2:
             image = image[:, :, None]
             
        # Normalize label to binary 0/1 for finding indices
        if label.max() > 1:
             label = label / 255.0

        h, w, c = image.shape
        th, tw = self.output_size[1], self.output_size[0]

        # 1. Pad if image is smaller than target
        if w < tw or h < th:
            pad_w = max(0, tw - w)
            pad_h = max(0, th - h)
            # Pad image (H, W, C)
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
            # Pad label (H, W) or (H, W, 1)? Usually label is (H, W)
            if len(label.shape) == 2:
                 label = np.pad(label, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            else:
                 label = np.pad(label, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
            h, w = label.shape[:2] # Update shape

        # 2. Find cracks
        # Use a small threshold to find crack pixels (assuming label is float or binary)
        y_inds, x_inds = np.where(label > 0)

        if len(y_inds) > 0:
            # Smart Crop: Center on a random crack pixel
            idx = random.randint(0, len(y_inds) - 1)
            cy, cx = y_inds[idx], x_inds[idx]

            # Determine crop coordinates (try to center the crack)
            y1 = max(0, cy - th // 2)
            x1 = max(0, cx - tw // 2)

            # Adjust if out of bounds (Shift back if near right/bottom edge)
            y1 = min(y1, h - th)
            x1 = min(x1, w - tw)
            
            # Final safety clip (shouldn't happen due to logic above but safe)
            y1 = int(max(0, y1))
            x1 = int(max(0, x1))
        else:
            # Fallback for images without cracks (rare): Random Crop
            y1 = random.randint(0, max(0, h - th))
            x1 = random.randint(0, max(0, w - tw))

        # Perform the Crop
        image = image[y1:y1+th, x1:x1+tw, :]
        if len(label.shape) == 2:
            label = label[y1:y1+th, x1:x1+tw]
        else:
            label = label[y1:y1+th, x1:x1+tw, :]
            
        # Convert to uint8 for Albumentations
        if image.dtype == np.float32 or image.dtype == np.float64:
             if image.max() <= 1.0:
                 image_uint8 = (image * 255).astype(np.uint8)
             else:
                 image_uint8 = image.astype(np.uint8)
        else:
             image_uint8 = image.astype(np.uint8)
             
        label_uint8 = (label > 0).astype(np.uint8)
        
        # Apply Albumentations (No Crop, only distortions)
        augmented = self.transform(image=image_uint8, mask=label_uint8)
        image_aug = augmented['image']
        label_aug = augmented['mask']
        
        # Convert back
        image = image_aug.astype(np.float32) / 255.0
        label = label_aug.astype(np.float32)

        
        if len(image.shape) == 3:
            x, y, c = image.shape
        else:
            x, y = image.shape
            c = 1

        # Resize with padding logic
        target_h, target_w = self.output_size[0], self.output_size[1]
        scale = min(target_w / y, target_h / x)
        new_w = int(y * scale)
        new_h = int(x * scale)

        # Skip resize if already correct (rare with float scale, but good practice)
        if new_w != y or new_h != x:
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            if c == 1 and len(image.shape) == 2:
                image = image[:, :, None]
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Create padded buffers
        final_image = np.zeros((target_h, target_w, c), dtype=np.float32)
        final_label = np.zeros((target_h, target_w), dtype=np.float32)

        # Center paste
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        
        if c > 1:
            final_image[pad_top:pad_top+new_h, pad_left:pad_left+new_w, :] = image
        else:
            final_image[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = image
            
        final_label[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = label

        # Clip values to [0, 1] to avoid cubic interpolation overshoot
        final_image = np.clip(final_image, 0, 1)

        image = torch.from_numpy(final_image)
        label = torch.from_numpy(final_label)  
        sample = {'image': image, 'label': label > 0.5} 

        return sample

class ValGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # Ensure numpy
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if isinstance(label, torch.Tensor):
            label = label.numpy()
            
        # --- Smart Crop Implementation (Deterministic for Val) ---
        if len(image.shape) == 2:
             image = image[:, :, None]
        if label.max() > 1:
             label = label / 255.0

        h, w, c = image.shape
        th, tw = self.output_size[1], self.output_size[0]

        # Pad
        if w < tw or h < th:
            pad_w = max(0, tw - w)
            pad_h = max(0, th - h)
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
            if len(label.shape) == 2:
                 label = np.pad(label, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            else:
                 label = np.pad(label, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
            h, w = label.shape[:2]

        # Find cracks
        y_inds, x_inds = np.where(label > 0)

        if len(y_inds) > 0:
            # Deterministic Center: Median
            cy = int(np.median(y_inds))
            cx = int(np.median(x_inds))

            y1 = max(0, cy - th // 2)
            x1 = max(0, cx - tw // 2)
            y1 = min(y1, h - th)
            x1 = min(x1, w - tw)
            y1 = int(max(0, y1))
            x1 = int(max(0, x1))
        else:
            # Center Crop fallback
            y1 = (h - th) // 2
            x1 = (w - tw) // 2
            y1 = max(0, y1)
            x1 = max(0, x1)

        # Crop
        image = image[y1:y1+th, x1:x1+tw, :]
        if len(label.shape) == 2:
            label = label[y1:y1+th, x1:x1+tw]
        else:
            label = label[y1:y1+th, x1:x1+tw, :]
            
        # Convert back
        if image.dtype == np.uint8:
             image = image.astype(np.float32) / 255.0
        label = label.astype(np.float32)

        # Clip just in case
        image = np.clip(image, 0, 1)

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        sample = {'image': image, 'label': label > 0.5}
        return sample
class GenericDataset(Dataset):
    def __init__(self, base_dir, split="train", transform=None, img_exts=None, mask_exts=None, output_size=None, cache_data=False):
        self.transform = transform  
        self.split = split
        self.data_dir = base_dir
        self.output_size = output_size # Tuple (h, w) or list
        self.cache_data = cache_data
        
        self.img_dir = os.path.join(base_dir, "images")
        self.mask_dir = os.path.join(base_dir, "masks")
        
        if not os.path.exists(self.img_dir) or not os.path.exists(self.mask_dir):
             raise ValueError(f"Dataset directory '{base_dir}' must contain 'images' and 'masks' subdirectories.")

        if img_exts is None:
            img_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

        self.sample_list = []
        for f in os.listdir(self.img_dir):
            if os.path.splitext(f)[1].lower() in img_exts:
                self.sample_list.append(f)
        
        self.sample_list.sort() # Ensure deterministic order
        print(f"GenericDataset ({split}): Found {len(self.sample_list)} images in {self.img_dir}")

        self.cached_images = {}
        self.cached_masks = {}

        if self.cache_data:
            print(f"GenericDataset ({split}): Caching data into RAM... This may take a while.")
            from concurrent.futures import ThreadPoolExecutor
            from tqdm import tqdm
            
            def load_single_item(idx):
                img_name = self.sample_list[idx]
                filepath_image = os.path.join(self.img_dir, img_name)
                base_name = os.path.splitext(img_name)[0]
                
                # Find mask path
                mask_path = None
                for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                    candidate = os.path.join(self.mask_dir, base_name + ext)
                    if os.path.exists(candidate):
                        mask_path = candidate
                        break
                
                if mask_path:
                    # Load and convert immediately to save space/time later
                    img = Image.open(filepath_image).convert("RGB")
                    dataset_mask = Image.open(mask_path).convert("L")
                    
                    # Store as numpy/bytes or keep as PIL? PIL is compact.
                    # Or convert to array now? Converting to array may take more RAM but faster access.
                    # Given 256GB RAM, let's store as numpy uint8 (compact visually).
                    return idx, np.array(img), np.array(dataset_mask)
                return idx, None, None

            with ThreadPoolExecutor(max_workers=16) as executor:
                results = list(tqdm(executor.map(load_single_item, range(len(self.sample_list))), total=len(self.sample_list)))
            
            for idx, img, mask in results:
                if img is not None:
                    self.cached_images[idx] = img
                    self.cached_masks[idx] = mask
            
            print(f"GenericDataset ({split}): Cached {len(self.cached_images)} items into RAM.")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.cache_data and idx in self.cached_images:
            image_arr = self.cached_images[idx]
            label_arr = self.cached_masks[idx]
            # Norm
            image = (image_arr / 255.0).astype(np.float32)
            label = (label_arr / 255.0).astype(np.float32)
        else:
            # Fallback to disk load (or if cache disabled)
            img_name = self.sample_list[idx]
            filepath_image = os.path.join(self.img_dir, img_name)
            
            base_name = os.path.splitext(img_name)[0]
            mask_name = None
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                # Try exact name match
                candidate = os.path.join(self.mask_dir, base_name + ext)
                if os.path.exists(candidate):
                    mask_name = base_name + ext
                    filepath_label = candidate
                    break
                
                # Try generic _mask suffix (common in crack datasets)
                candidate_mask = os.path.join(self.mask_dir, base_name + "_mask" + ext)
                if os.path.exists(candidate_mask):
                    mask_name = base_name + "_mask" + ext
                    filepath_label = candidate_mask
                    break
            
            if mask_name is None:
                 raise FileNotFoundError(f"No corresponding mask found for image {img_name} in {self.mask_dir}")
    
            image = Image.open(filepath_image).convert("RGB")
            label = Image.open(filepath_label).convert("L")
    
            image = (np.array(image) / 255.0).astype(np.float32)
            label = (np.array(label) / 255.0).astype(np.float32)

        sample = {'image': image, 'label': label}

        # Apply transformation (Augmentation) if any
        if self.transform:
            sample = self.transform(sample)
            image, label = sample['image'], sample['label']
            if isinstance(image, torch.Tensor):
                 image = image.numpy()
            if isinstance(label, torch.Tensor):
                 label = label.numpy()
        else:
             image, label = sample['image'], sample['label']
        
        # Apply deterministic resize with padding if output_size is set
        if self.output_size and not self.transform: 
             if len(image.shape) == 3:
                x, y, c = image.shape
             else:
                x, y = image.shape
                c = 1

             target_h, target_w = self.output_size[0], self.output_size[1]
             scale = min(target_w / y, target_h / x)
             new_w = int(y * scale)
             new_h = int(x * scale)

             if new_w != y or new_h != x:
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                if c == 1 and len(image.shape) == 2:
                    image = image[:, :, None]
                label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

             final_image = np.zeros((target_h, target_w, c), dtype=np.float32)
             final_label = np.zeros((target_h, target_w), dtype=np.float32)

             pad_top = (target_h - new_h) // 2
             pad_left = (target_w - new_w) // 2
        
             if c > 1:
                final_image[pad_top:pad_top+new_h, pad_left:pad_left+new_w, :] = image
             else:
                final_image[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = image
            
             final_label[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = label
             
             final_image = np.clip(final_image, 0, 1)

             image = torch.from_numpy(final_image)
             label = torch.from_numpy(final_label)
        
        if not isinstance(image, torch.Tensor):
             image = torch.from_numpy(image.astype(np.float32)) 
             label = torch.from_numpy(label.astype(np.float32))

        # Standardize format: C, H, W
        if image.ndim == 3 and image.shape[2] == 3: # H, W, C -> C, H, W
             image = image.permute(2, 0, 1)
        elif image.ndim == 2:
             # Add channel dim 1, H, W? Or just H, W?
             # Model expects B, C, H, W usually.
             pass
        
        # Ensure label is binary for validation/training if needed, or keep float
        # Label should be (H, W) or (1, H, W)?
        # Usually label is (H, W).
        if isinstance(label, torch.Tensor):
             label = label > 0.5

        # --- Prompt Engineering (Box & Point Generation) ---
        # Convert label back to numpy for processing
        if isinstance(label, torch.Tensor):
            mask_np = label.numpy().astype(np.uint8)
        else:
            mask_np = label.astype(np.uint8)

        # Find bounding box
        coords = np.argwhere(mask_np > 0)
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            # Add some jitter/padding safely
            h, w = mask_np.shape
            if self.split == 'train':
                pad_x1 = np.random.randint(0, 10)
                pad_y1 = np.random.randint(0, 10)
                pad_x2 = np.random.randint(0, 10)
                pad_y2 = np.random.randint(0, 10)
            else:
                # Deterministic padding for Val/Test
                pad_x1 = 5
                pad_y1 = 5
                pad_x2 = 5
                pad_y2 = 5

            x_min = max(0, x_min - pad_x1)
            y_min = max(0, y_min - pad_y1)
            x_max = min(w, x_max + pad_x2)
            y_max = min(h, y_max + pad_y2)
            box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

            # Point prompt: 1 positive (center of random crack part) + 1 random negative
            # Positive point
            pos_indices = np.argwhere(mask_np > 0)
            if self.split == 'train':
                pos_pt = pos_indices[np.random.randint(len(pos_indices))] # [y, x]
            else:
                # Deterministic point (Median/Center)
                # Sort indices to ensure deterministic order then pick middle
                # argwhere returns sorted order usually logic wise (row major), but safe to just pick middle
                pos_pt = pos_indices[len(pos_indices) // 2]
            pos_pt = pos_pt[::-1] # [x, y]

        else:
            # Empty mask: Box is full image, no pos point
            h, w = mask_np.shape
            box = np.array([0, 0, w, h], dtype=np.float32)
            pos_pt = np.array([w//2, h//2]) # Dummy center

        # Negative point (background)
        neg_indices = np.argwhere(mask_np == 0)
        if len(neg_indices) > 0:
            if self.split == 'train':
                neg_pt = neg_indices[np.random.randint(len(neg_indices))]
            else:
                # Deterministic negative point
                neg_pt = neg_indices[len(neg_indices) // 2]
            neg_pt = neg_pt[::-1] # [x, y]
        else:
            neg_pt = np.array([0, 0])

        point_coords = np.array([pos_pt, neg_pt], dtype=np.float32)
        point_labels = np.array([1, 0], dtype=np.float32) # 1=pos, 0=neg

        sample = {
            'image': image, 
            'label': label, 
            'box': torch.from_numpy(box),
            'point_coords': torch.from_numpy(point_coords),
            'point_labels': torch.from_numpy(point_labels),
            'case_name': base_name
        }
        return sample

