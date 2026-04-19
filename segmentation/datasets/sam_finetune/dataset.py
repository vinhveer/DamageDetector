import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image
from torch_runtime import Dataset, torch
from tqdm import tqdm

from ..core import DEFAULT_MASK_EXTS, build_crop_metadata, find_mask_path, get_mask_index, list_image_files
from .prompts import build_prompt_tensors


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class GenericDataset(Dataset):
    def __init__(
        self,
        base_dir,
        split="train",
        transform=None,
        img_exts=None,
        mask_exts=None,
        output_size=None,
        cache_data=False,
        patches_per_image=1,
        use_full_image_box: bool = False,
    ):
        self.transform = transform
        self.split = split
        self.data_dir = base_dir
        self.output_size = output_size
        self.cache_data = cache_data
        self.patches_per_image = patches_per_image
        self.use_full_image_box = bool(use_full_image_box)

        self.img_dir = os.path.join(base_dir, "images")
        self.mask_dir = os.path.join(base_dir, "masks")

        if not os.path.exists(self.img_dir) or not os.path.exists(self.mask_dir):
            raise ValueError(f"Dataset directory '{base_dir}' must contain 'images' and 'masks' subdirectories.")

        if img_exts is None:
            img_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
        if mask_exts is None:
            mask_exts = list(DEFAULT_MASK_EXTS)
        self.mask_exts = tuple(str(ext).lower() for ext in mask_exts)

        self.sample_list = list_image_files(self.img_dir, img_exts=img_exts)
        self.mask_index = get_mask_index(self.mask_dir, mask_exts=self.mask_exts)
        print(f"GenericDataset ({split}): Found {len(self.sample_list)} images in {self.img_dir}")

        self.cached_images = {}
        self.cached_masks = {}
        self.cached_crop_metadata = {}

        if self.cache_data:
            print(f"GenericDataset ({split}): Caching data into RAM... This may take a while.")

            def load_single_item(idx):
                img_name = self.sample_list[idx]
                filepath_image = os.path.join(self.img_dir, img_name)
                base_name = os.path.splitext(img_name)[0]
                mask_path = find_mask_path(self.mask_dir, base_name, mask_index=self.mask_index)
                if mask_path:
                    img = Image.open(filepath_image).convert("RGB")
                    dataset_mask = Image.open(mask_path).convert("L")
                    image_arr = np.array(img)
                    mask_arr = np.array(dataset_mask)
                    crop_metadata = build_crop_metadata(image_arr, mask_arr, crop_policy=getattr(self.transform, "crop_policy", "smart"))
                    return idx, image_arr, mask_arr, crop_metadata
                return idx, None, None, None

            with ThreadPoolExecutor(max_workers=16) as executor:
                results = list(tqdm(executor.map(load_single_item, range(len(self.sample_list))), total=len(self.sample_list)))

            for idx, img, mask, crop_metadata in results:
                if img is not None:
                    self.cached_images[idx] = img
                    self.cached_masks[idx] = mask
                    self.cached_crop_metadata[idx] = crop_metadata

            print(f"GenericDataset ({split}): Cached {len(self.cached_images)} items into RAM.")

    def __len__(self):
        return len(self.sample_list) * self.patches_per_image

    def __getitem__(self, idx):
        idx = idx // self.patches_per_image
        img_name = self.sample_list[idx]
        base_name = os.path.splitext(img_name)[0]

        if self.cache_data and idx in self.cached_images:
            image_arr = self.cached_images[idx]
            label_arr = self.cached_masks[idx]
            crop_metadata = self.cached_crop_metadata.get(idx)
            image = image_arr.astype(np.float32)
            label = (label_arr / 255.0).astype(np.float32)
        else:
            filepath_image = os.path.join(self.img_dir, img_name)
            filepath_label = find_mask_path(self.mask_dir, base_name, mask_index=self.mask_index)
            if filepath_label is None:
                raise FileNotFoundError(f"No corresponding mask found for image {img_name} in {self.mask_dir}")

            image = np.array(Image.open(filepath_image).convert("RGB")).astype(np.float32)
            label = (np.array(Image.open(filepath_label).convert("L")) / 255.0).astype(np.float32)
            crop_metadata = build_crop_metadata(image, label, crop_policy=getattr(self.transform, "crop_policy", "smart"))

        sample = {"image": image, "label": label, "crop_metadata": crop_metadata}

        if self.transform:
            sample = self.transform(sample)
            image, label = sample["image"], sample["label"]
            if isinstance(image, torch.Tensor):
                image = image.numpy()
            if isinstance(label, torch.Tensor):
                label = label.numpy()
        else:
            image, label = sample["image"], sample["label"]

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
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                if c == 1 and len(image.shape) == 2:
                    image = image[:, :, None]
                label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            final_image = np.zeros((target_h, target_w, c), dtype=np.float32)
            final_label = np.zeros((target_h, target_w), dtype=np.float32)
            pad_top = (target_h - new_h) // 2
            pad_left = (target_w - new_w) // 2

            if c > 1:
                final_image[pad_top:pad_top + new_h, pad_left:pad_left + new_w, :] = image
            else:
                final_image[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = image
            final_label[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = label
            final_image = np.clip(final_image, 0, 1)

            image = torch.from_numpy(final_image)
            label = torch.from_numpy(final_label)

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.astype(np.float32))
            label = torch.from_numpy(label.astype(np.float32))

        if image.ndim == 3 and image.shape[2] == 3:
            image = image.permute(2, 0, 1)

        if isinstance(label, torch.Tensor):
            label = label > 0.5

        if isinstance(label, torch.Tensor):
            mask_np = label.numpy().astype(np.uint8)
        else:
            mask_np = label.astype(np.uint8)

        sample = {
            "image": image,
            "label": label,
            **build_prompt_tensors(
                mask_np,
                split=self.split,
                use_full_image_box=self.use_full_image_box,
                case_name=base_name,
            ),
        }
        return sample
