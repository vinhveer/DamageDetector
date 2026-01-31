import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image
import glob

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
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
            # Zoom is still useful for resizing but we control the factors to keep aspect ratio
            # Warning: zoom takes factors, not sizes.
            # Factor = new / old
            image = zoom(image, (new_h / x, new_w / y, 1) if c > 1 else (new_h / x, new_w / y), order=3)
            label = zoom(label, (new_h / x, new_w / y), order=0)

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
    def __init__(self, base_dir, split="train", transform=None, img_exts=None, mask_exts=None, output_size=None):
        self.transform = transform  
        self.split = split
        self.data_dir = base_dir
        self.output_size = output_size # Tuple (h, w) or list
        
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

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
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

        # Load
        image = Image.open(filepath_image).convert("RGB") # Ensure RGB
        label = Image.open(filepath_label).convert("L")   # Ensure Grayscale

        image = np.array(image) / 255.0  
        label = np.array(label) / 255.0    

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
                image = zoom(image, (new_h / x, new_w / y, 1) if c > 1 else (new_h / x, new_w / y), order=3)
                label = zoom(label, (new_h / x, new_w / y), order=0)

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

        sample = {'image': image, 'label': label, 'case_name': base_name}
        return sample

