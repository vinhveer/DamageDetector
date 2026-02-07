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
        # Define Albumentations pipeline
        # Define Albumentations pipeline (Heavy Augmentation)
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            
            # Geometric Augmentations
            # Safe Logic: Pad -> RandomCrop
            A.PadIfNeeded(min_height=output_size[1], min_width=output_size[0], border_mode=0, value=0, mask_value=0),
            
            # NOTE: SAM originally uses RandomResizedCrop (scale/aspect ratio change).
            # If we want exact pixel learning like we did for UNet, we use RandomCrop.
            # But SAM benefits from scale variance. 
            # However, to fix the crash "crop > image size", we MUST Pad first.
            # If we still want scale variance, we can keep RandomResizedCrop BUT after Padding it's safer?
            # Actually RandomResizedCrop on Padded image is safe.
            # BUT user asked "Sửa luôn cho bên sam finetune" implying adopt the RandomCrop strategy.
            # Let's switch to RandomCrop for consistency and safety.
            A.RandomCrop(height=output_size[1], width=output_size[0], p=1.0),
            A.Affine(scale=(0.9, 1.1), translate_percent=0.0625, rotate=30, p=0.5),
            A.OneOf([
                A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
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
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.3),
            A.CoarseDropout(num_holes_limit=(1, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), min_holes=None, min_height=None, min_width=None, fill_value=0, mask_fill_value=0, p=0.3),
        ], is_check_shapes=False)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # Apply Albumentations
        # Convert float image (0-1) to uint8 (0-255) for optimal Albumentations support
        if image.dtype == np.float32 or image.dtype == np.float64:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image

        # Ensure mask is uint8 (required by some transforms like ElasticTransform)
        label_uint8 = label.astype(np.uint8)
        
        augmented = self.transform(image=image_uint8, mask=label_uint8)
        image_aug = augmented['image']
        label_aug = augmented['mask']
        
        # Convert back to float 0-1
        image = image_aug.astype(np.float32) / 255.0
        label = label_aug # Keep as is (0,1 or 0,255 depending on input, but mask is usually binary)
        
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
                candidate = os.path.join(self.mask_dir, base_name + ext)
                if os.path.exists(candidate):
                    mask_name = base_name + ext
                    filepath_label = candidate
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
            x_min = max(0, x_min - np.random.randint(0, 10))
            y_min = max(0, y_min - np.random.randint(0, 10))
            x_max = min(w, x_max + np.random.randint(0, 10))
            y_max = min(h, y_max + np.random.randint(0, 10))
            box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

            # Point prompt: 1 positive (center of random crack part) + 1 random negative
            # Positive point
            pos_indices = np.argwhere(mask_np > 0)
            pos_pt = pos_indices[np.random.randint(len(pos_indices))] # [y, x]
            pos_pt = pos_pt[::-1] # [x, y]

        else:
            # Empty mask: Box is full image, no pos point
            h, w = mask_np.shape
            box = np.array([0, 0, w, h], dtype=np.float32)
            pos_pt = np.array([w//2, h//2]) # Dummy center

        # Negative point (background)
        neg_indices = np.argwhere(mask_np == 0)
        if len(neg_indices) > 0:
            neg_pt = neg_indices[np.random.randint(len(neg_indices))]
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

