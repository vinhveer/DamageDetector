import os
import random
import numpy as np
from torch_runtime import torch

from torch_runtime import Dataset
from PIL import Image
import glob
import cv2
import albumentations as A

# Prevent OpenCV from using multithreading within workers
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

DEFAULT_MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_MASK_INDEX_CACHE: dict[tuple[str, tuple[str, ...]], dict[str, str]] = {}

def random_rot_flip(image, label):
    # Deprecated: Handled by Albumentations
    pass

def random_rotate(image, label):
    # Deprecated: Handled by Albumentations
    pass


def build_mask_index(mask_dir: str, mask_exts=None) -> dict[str, str]:
    ext_values = tuple(str(ext).lower() for ext in (mask_exts or DEFAULT_MASK_EXTS))
    index: dict[str, str] = {}
    if not os.path.isdir(mask_dir):
        return index

    def _sort_key(name: str) -> tuple[int, str]:
        ext = os.path.splitext(name)[1].lower()
        try:
            ext_rank = ext_values.index(ext)
        except ValueError:
            ext_rank = len(ext_values)
        return ext_rank, name

    for file_name in sorted(os.listdir(mask_dir), key=_sort_key):
        stem, ext = os.path.splitext(file_name)
        if ext.lower() not in ext_values:
            continue
        full_path = os.path.join(mask_dir, file_name)
        if not os.path.isfile(full_path):
            continue
        candidates = [stem]
        if stem.endswith("_mask"):
            candidates.append(stem[:-5])
        for key in candidates:
            if key and key not in index:
                index[key] = full_path
    return index


def get_mask_index(mask_dir: str, mask_exts=None) -> dict[str, str]:
    ext_values = tuple(str(ext).lower() for ext in (mask_exts or DEFAULT_MASK_EXTS))
    cache_key = (os.path.abspath(mask_dir), ext_values)
    mask_index = _MASK_INDEX_CACHE.get(cache_key)
    if mask_index is None:
        mask_index = build_mask_index(mask_dir, mask_exts=ext_values)
        _MASK_INDEX_CACHE[cache_key] = mask_index
    return mask_index


def find_mask_path(mask_dir: str, base_name: str, mask_index: dict[str, str] | None = None, mask_exts=None) -> str | None:
    index = mask_index if mask_index is not None else get_mask_index(mask_dir, mask_exts=mask_exts)
    return index.get(str(base_name))


def list_image_files(image_dir: str, img_exts=None) -> list[str]:
    if img_exts is None:
        img_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    names = []
    for f in os.listdir(image_dir):
        if os.path.splitext(f)[1].lower() in img_exts:
            names.append(f)
    names.sort()
    return names


def load_image_mask_arrays(base_dir: str, image_name: str) -> tuple[np.ndarray, np.ndarray]:
    img_dir = os.path.join(base_dir, "images")
    mask_dir = os.path.join(base_dir, "masks")
    image_path = os.path.join(img_dir, image_name)
    base_name = os.path.splitext(image_name)[0]
    mask_path = find_mask_path(mask_dir, base_name, mask_index=get_mask_index(mask_dir))
    if mask_path is None:
        raise FileNotFoundError(f"No corresponding mask found for image {image_name} in {mask_dir}")
    image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    mask = (np.array(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0).astype(np.uint8)
    return image, mask


def _select_point(coords: np.ndarray, *, split: str) -> np.ndarray:
    if len(coords) == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    if split == "train":
        point = coords[np.random.randint(len(coords))]
    else:
        point = coords[len(coords) // 2]
    return point[::-1].astype(np.float32)


def _sample_negative_point(mask_np: np.ndarray, *, split: str) -> np.ndarray:
    ring_coords = np.empty((0, 2), dtype=np.int32)
    if int(mask_np.sum()) > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated = cv2.dilate(mask_np.astype(np.uint8), kernel, iterations=1)
        ring = np.logical_and(dilated > 0, mask_np == 0)
        ring_coords = np.argwhere(ring)
    if len(ring_coords) > 0:
        return _select_point(ring_coords, split=split)

    bg_coords = np.argwhere(mask_np == 0)
    if len(bg_coords) > 0:
        return _select_point(bg_coords, split=split)

    h_img, w_img = mask_np.shape
    return np.array([float(w_img // 2), float(h_img // 2)], dtype=np.float32)

class RandomGenerator(object):
    def __init__(
        self,
        output_size,
        low_res,
        background_crop_prob: float = 0.2,
        near_background_crop_prob: float = 0.15,
        hard_negative_crop_prob: float = 0.10,
        augment_profile: str = "balanced",
        crop_policy: str = "smart",
    ):
        self.output_size = output_size
        self.low_res = low_res
        self.background_crop_prob = float(background_crop_prob)
        self.near_background_crop_prob = float(near_background_crop_prob)
        self.hard_negative_crop_prob = float(hard_negative_crop_prob)
        self.augment_profile = str(augment_profile or "balanced").strip().lower()
        self.crop_policy = str(crop_policy or "smart").strip().lower()
        self.transform = self._build_transform(self.augment_profile)

    def _build_transform(self, augment_profile: str):
        profile = str(augment_profile or "balanced").strip().lower()
        if profile == "strong":
            profile = "aggressive"

        if profile == "aggressive":
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf([
                    A.Affine(
                        scale=(0.85, 1.15),
                        translate_percent=(-0.06, 0.06),
                        rotate=(-35, 35),
                        shear=(-6, 6),
                        interpolation=cv2.INTER_LINEAR,
                        mask_interpolation=cv2.INTER_NEAREST,
                        p=1.0,
                    ),
                    A.Perspective(scale=(0.02, 0.05), keep_size=True, fit_output=False, p=1.0),
                ], p=0.55),
                A.OneOf([
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                ], p=0.5),
                A.OneOf([
                    A.RandomShadow(shadow_roi=(0.0, 0.0, 1.0, 1.0), p=1.0),
                    A.RandomToneCurve(scale=0.15, p=1.0),
                ], p=0.25),
                A.OneOf([
                    A.ImageCompression(p=1.0),
                    A.Downscale(p=1.0),
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(p=1.0),
                    A.Blur(blur_limit=3, p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                ], p=0.35),
                A.CoarseDropout(p=0.2),
            ], is_check_shapes=False)

        if profile == "light":
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.35),
                A.Affine(
                    scale=(0.95, 1.05),
                    translate_percent=(-0.02, 0.02),
                    rotate=(-10, 10),
                    shear=(-2, 2),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    p=0.15,
                ),
                A.OneOf([
                    A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=1.0),
                    A.RandomGamma(gamma_limit=(92, 108), p=1.0),
                ], p=0.22),
                A.OneOf([
                    A.GaussNoise(p=1.0),
                    A.Blur(blur_limit=3, p=1.0),
                ], p=0.08),
            ], is_check_shapes=False)

        # Balanced profile: keep crack geometry sharper and reduce augmentations that
        # inflate robustness at the expense of boundary quality / IoU.
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.04, 0.04),
                    rotate=(-20, 20),
                    shear=(-4, 4),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    p=1.0,
                ),
                A.Perspective(scale=(0.01, 0.03), keep_size=True, fit_output=False, p=1.0),
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.16, contrast_limit=0.16, p=1.0),
                A.RandomGamma(gamma_limit=(85, 115), p=1.0),
            ], p=0.35),
            A.OneOf([
                A.RandomShadow(shadow_roi=(0.0, 0.0, 1.0, 1.0), p=1.0),
                A.RandomToneCurve(scale=0.12, p=1.0),
            ], p=0.12),
            A.OneOf([
                A.ImageCompression(p=1.0),
                A.Downscale(p=1.0),
            ], p=0.12),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.Blur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.18),
            A.CoarseDropout(p=0.08),
        ], is_check_shapes=False)

    def _random_crop_coords(self, h: int, w: int, th: int, tw: int) -> tuple[int, int]:
        y1 = random.randint(0, max(0, h - th))
        x1 = random.randint(0, max(0, w - tw))
        return y1, x1

    def _hard_negative_crop_coords(self, image: np.ndarray, label: np.ndarray, h: int, w: int, th: int, tw: int) -> tuple[int, int]:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY) if image.ndim == 3 and image.shape[2] == 3 else image.astype(np.uint8)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_map = cv2.magnitude(sobel_x, sobel_y)

        best_score = -1.0
        best_coords = self._random_crop_coords(h, w, th, tw)
        for _ in range(32):
            y1, x1 = self._random_crop_coords(h, w, th, tw)
            label_crop = label[y1:y1 + th, x1:x1 + tw]
            if int((label_crop > 0).sum()) > 0:
                continue
            edge_crop = edge_map[y1:y1 + th, x1:x1 + tw]
            score = float(edge_crop.mean())
            if score > best_score:
                best_score = score
                best_coords = (y1, x1)
        return best_coords

    def _choose_crop_coords(self, image: np.ndarray, label: np.ndarray, h: int, w: int, th: int, tw: int) -> tuple[int, int]:
        y_inds, x_inds = np.where(label > 0)

        if self.crop_policy == "fast":
            if len(y_inds) > 0:
                crop_mode = random.random()
                if crop_mode < self.background_crop_prob:
                    return self._random_crop_coords(h, w, th, tw)
                idx = random.randint(0, len(y_inds) - 1)
                cy, cx = y_inds[idx], x_inds[idx]
                offset_scale = 0.75 if crop_mode < self.background_crop_prob + self.near_background_crop_prob else 0.35
                cy += random.randint(-int(th * offset_scale), int(th * offset_scale))
                cx += random.randint(-int(tw * offset_scale), int(tw * offset_scale))
                y1 = max(0, min(h - th, cy - th // 2))
                x1 = max(0, min(w - tw, cx - tw // 2))
                return int(y1), int(x1)
            return self._random_crop_coords(h, w, th, tw)

        if len(y_inds) > 0:
            crop_mode = random.random()
            if crop_mode < self.background_crop_prob:
                y1, x1 = self._random_crop_coords(h, w, th, tw)
            elif crop_mode < self.background_crop_prob + self.hard_negative_crop_prob:
                y1, x1 = self._hard_negative_crop_coords(image, label, h, w, th, tw)
            else:
                idx = random.randint(0, len(y_inds) - 1)
                cy, cx = y_inds[idx], x_inds[idx]

                if crop_mode < self.background_crop_prob + self.hard_negative_crop_prob + self.near_background_crop_prob:
                    offset_y = random.randint(-int(th * 0.9), int(th * 0.9))
                    offset_x = random.randint(-int(tw * 0.9), int(tw * 0.9))
                else:
                    offset_y = random.randint(-int(th * 0.4), int(th * 0.4))
                    offset_x = random.randint(-int(tw * 0.4), int(tw * 0.4))

                cy += offset_y
                cx += offset_x

                y1 = max(0, cy - th // 2)
                x1 = max(0, cx - tw // 2)
                y1 = min(y1, h - th)
                x1 = min(x1, w - tw)
                y1 = int(max(0, y1))
                x1 = int(max(0, x1))
        else:
            if random.random() < self.hard_negative_crop_prob:
                y1, x1 = self._hard_negative_crop_coords(image, label, h, w, th, tw)
            else:
                y1, x1 = self._random_crop_coords(h, w, th, tw)
        return int(y1), int(x1)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # Ensure numpy
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if isinstance(label, torch.Tensor):
            label = label.numpy()
            
        if image.shape[:2] != label.shape[:2]:
            raise ValueError(
                f"Image/mask size mismatch: image={image.shape[:2]} mask={label.shape[:2]}. "
                "Fix the dataset instead of silently cropping/padding labels."
            )

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
            
            # Pad image (Use Zero Padding to avoid reflecting cracks!)
            # Reflection can duplicate a crack at the border, but the mask remains 0 (background),
            # confusing the model. Zero padding is safer for thin structures like cracks.
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            
            # Pad label (Keep explicit 0 for background/ignore)
            label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            
            h, w = label.shape[:2] # Update shape

        y1, x1 = self._choose_crop_coords(image, label, h, w, th, tw)

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
        image = image_aug.astype(np.float32)
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
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
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

        # FIX: Normalize to 0-1 before clipping
        final_image = final_image / 255.0
        # Clip values to [0, 1] to avoid cubic interpolation overshoot
        final_image = np.clip(final_image, 0, 1)

        image = torch.from_numpy(final_image)
        label = torch.from_numpy(final_label)  
        sample = {'image': image, 'label': label > 0.5} 

        return sample


class RefineRandomGenerator(RandomGenerator):
    def __init__(
        self,
        output_size,
        low_res,
        background_crop_prob: float = 0.1,
        near_background_crop_prob: float = 0.2,
        hard_negative_crop_prob: float = 0.1,
        augment_profile: str = "balanced",
        crop_policy: str = "smart",
        roi_positive_band_low: float = 0.20,
        roi_positive_band_high: float = 0.90,
    ):
        super().__init__(
            output_size=output_size,
            low_res=low_res,
            background_crop_prob=background_crop_prob,
            near_background_crop_prob=near_background_crop_prob,
            hard_negative_crop_prob=hard_negative_crop_prob,
            augment_profile=augment_profile,
            crop_policy=crop_policy,
        )
        self.roi_positive_band_low = float(roi_positive_band_low)
        self.roi_positive_band_high = float(roi_positive_band_high)

    def _choose_crop_coords(self, image: np.ndarray, label: np.ndarray, h: int, w: int, th: int, tw: int) -> tuple[int, int]:
        mask = (label > 0).astype(np.uint8)
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return super()._choose_crop_coords(image, label, h, w, th, tw)

        if self.crop_policy == "fast":
            point = coords[np.random.randint(len(coords))]
            cy, cx = int(point[0]), int(point[1])
            cy += random.randint(-int(th * 0.25), int(th * 0.25))
            cx += random.randint(-int(tw * 0.25), int(tw * 0.25))
            y1 = max(0, min(h - th, cy - th // 2))
            x1 = max(0, min(w - tw, cx - tw // 2))
            return int(y1), int(x1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, kernel, iterations=1)
        boundary = np.logical_and(dilated > 0, eroded == 0)
        ambiguous_bg = np.logical_and(dilated > 0, mask == 0)

        mode = random.random()
        if mode < 0.45:
            selected = coords
            offset_scale = 0.20
        elif mode < 0.75 and int(boundary.sum()) > 0:
            selected = np.argwhere(boundary)
            offset_scale = 0.15
        elif int(ambiguous_bg.sum()) > 0:
            selected = np.argwhere(ambiguous_bg)
            offset_scale = 0.25
        else:
            selected = coords
            offset_scale = 0.20

        point = selected[np.random.randint(len(selected))]
        cy, cx = int(point[0]), int(point[1])
        cy += random.randint(-int(th * offset_scale), int(th * offset_scale))
        cx += random.randint(-int(tw * offset_scale), int(tw * offset_scale))
        y1 = max(0, min(h - th, cy - th // 2))
        x1 = max(0, min(w - tw, cx - tw // 2))
        return int(y1), int(x1)

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
            
        if image.shape[:2] != label.shape[:2]:
            raise ValueError(
                f"Image/mask size mismatch: image={image.shape[:2]} mask={label.shape[:2]}. "
                "Fix the dataset instead of silently cropping/padding labels."
            )

        # --- Smart Crop Implementation (Deterministic for Val) ---
        if len(image.shape) == 2:
             image = image[:, :, None]
        if label.max() > 1:
             label = label / 255.0

        h, w, c = image.shape
        th, tw = self.output_size[1], self.output_size[0]

        # Pad
        # Pad
        if w < tw or h < th:
            pad_w = max(0, tw - w)
            pad_h = max(0, th - h)
            
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            
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
             image = image.astype(np.float32)
        label = label.astype(np.float32)

        # Clip just in case
        image = image / 255.0
        image = np.clip(image, 0, 1)

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        sample = {'image': image, 'label': label > 0.5}
        return sample
class GenericDataset(Dataset):
    def __init__(self, base_dir, split="train", transform=None, img_exts=None, mask_exts=None, output_size=None,
                 cache_data=False, patches_per_image=1, use_full_image_box: bool = False):
        self.transform = transform  
        self.split = split
        self.data_dir = base_dir
        self.output_size = output_size # Tuple (h, w) or list
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

        if self.cache_data:
            print(f"GenericDataset ({split}): Caching data into RAM... This may take a while.")
            from concurrent.futures import ThreadPoolExecutor
            from tqdm import tqdm
            
            def load_single_item(idx):
                img_name = self.sample_list[idx]
                filepath_image = os.path.join(self.img_dir, img_name)
                base_name = os.path.splitext(img_name)[0]
                
                mask_path = find_mask_path(self.mask_dir, base_name, mask_index=self.mask_index)
                
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
        return len(self.sample_list) * self.patches_per_image

    def __getitem__(self, idx):
        # Map augmented index to original image index
        idx = idx // self.patches_per_image
        img_name = self.sample_list[idx]
        base_name = os.path.splitext(img_name)[0]
        
        if self.cache_data and idx in self.cached_images:
            image_arr = self.cached_images[idx]
            label_arr = self.cached_masks[idx]
            # Norm
            image = (image_arr).astype(np.float32)
            label = (label_arr / 255.0).astype(np.float32)
        else:
            # Fallback to disk load (or if cache disabled)
            filepath_image = os.path.join(self.img_dir, img_name)
            filepath_label = find_mask_path(self.mask_dir, base_name, mask_index=self.mask_index)
            if filepath_label is None:
                 raise FileNotFoundError(f"No corresponding mask found for image {img_name} in {self.mask_dir}")
    
            image = Image.open(filepath_image).convert("RGB")
            label = Image.open(filepath_label).convert("L")
    
            image = (np.array(image)).astype(np.float32)
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
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
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

        # Build prompt candidates from the transformed mask.
        coords = np.argwhere(mask_np > 0)
        h, w = mask_np.shape
        has_foreground = len(coords) > 0
        full_box = np.array([0, 0, w, h], dtype=np.float32)

        if has_foreground:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
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
            tight_box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
            pos_pt = _select_point(coords, split=self.split)
        else:
            tight_box = full_box.copy()
            pos_pt = np.array([float(w // 2), float(h // 2)], dtype=np.float32)

        neg_pt = _sample_negative_point(mask_np, split=self.split)
        box = full_box if self.use_full_image_box or not has_foreground else tight_box
        if has_foreground:
            point_coords = np.array([pos_pt, neg_pt], dtype=np.float32)
            point_labels = np.array([1, 0], dtype=np.float32)
        else:
            point_coords = np.array([neg_pt, neg_pt], dtype=np.float32)
            point_labels = np.array([0, 0], dtype=np.float32)

        sample = {
            'image': image, 
            'label': label, 
            'box': torch.from_numpy(box),
            'full_box': torch.from_numpy(full_box),
            'tight_box': torch.from_numpy(tight_box),
            'pos_point': torch.from_numpy(pos_pt),
            'neg_point': torch.from_numpy(neg_pt),
            'has_foreground': torch.tensor(bool(has_foreground)),
            'point_coords': torch.from_numpy(point_coords),
            'point_labels': torch.from_numpy(point_labels),
            'case_name': base_name
        }
        return sample
