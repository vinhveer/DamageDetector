import os
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

"""
Data pipeline utilities:
- Crack dataset loading, preprocessing, and augmentation.
- Attention Gate module used by the U-Net decoder.
"""


class LetterboxResize:
    """
    Resize while preserving aspect ratio, then pad to the target size.

    This avoids geometric distortion from Resize((H, W)) on non-square images.

    Notes:
    - For masks, use nearest-neighbor interpolation to avoid creating gray edges.
    - For RGB images, use bilinear/bicubic interpolation for smoother results.
    """

    def __init__(self, size, fill=0, interpolation=Image.BILINEAR):
        if isinstance(size, int):
            self.target_w = size
            self.target_h = size
        else:
            self.target_h, self.target_w = size  # torchvision convention (H, W)
        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == 0 or h == 0:
            return img

        scale = min(self.target_w / w, self.target_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized = img.resize((new_w, new_h), resample=self.interpolation)

        if img.mode == "RGB":
            fill = self.fill if isinstance(self.fill, tuple) else (self.fill, self.fill, self.fill)
        else:
            fill = self.fill if not isinstance(self.fill, tuple) else self.fill[0]

        canvas = Image.new(img.mode, (self.target_w, self.target_h), color=fill)
        left = (self.target_w - new_w) // 2
        top = (self.target_h - new_h) // 2
        canvas.paste(resized, (left, top))
        return canvas


def _list_valid_images(image_dir, mask_dir, mask_prefix="auto"):
    images = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    )
    valid = []
    for img in images:
        base_name = os.path.splitext(img)[0]
        mask_path = find_mask_path(mask_dir, base_name, mask_prefix=mask_prefix)
        if mask_path is not None:
            valid.append(img)
    return valid


def find_mask_path(mask_dir: str, image_base_name: str, mask_prefix: str = "auto"):
    """
    Find a mask file by base name + optional suffix, allowing any extension.

    Example:
      image 'abc.jpg' -> image_base_name='abc'
      mask_prefix='_mask' => look for 'abc_mask.*'
      mask_prefix='' => look for 'abc.*'
      mask_prefix='auto' => try both '' and '_mask'
    """
    if mask_prefix is None:
        prefixes = ["_mask", ""]
    else:
        mask_prefix = str(mask_prefix)
        if mask_prefix.lower() == "auto":
            prefixes = ["_mask", ""]
        else:
            prefixes = [mask_prefix]

    try:
        names = os.listdir(mask_dir)
    except FileNotFoundError:
        return None

    preferred_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
    for prefix in prefixes:
        target_stem = f"{image_base_name}{prefix}"
        candidates = []
        for name in names:
            stem, _ext = os.path.splitext(name)
            if stem == target_stem:
                candidates.append(name)

        if not candidates:
            continue

        candidates_sorted = sorted(candidates)
        for ext in preferred_exts:
            for name in candidates_sorted:
                if os.path.splitext(name)[1].lower() == ext:
                    return os.path.join(mask_dir, name)
        return os.path.join(mask_dir, candidates_sorted[0])
    return None


def _normalize_patch_size(patch_size):
    if patch_size is None:
        return None
    if isinstance(patch_size, int):
        return (patch_size, patch_size)
    if isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
        return (int(patch_size[0]), int(patch_size[1]))
    raise ValueError(f"Invalid patch_size: {patch_size!r}")


def _pad_to_min_size(img: Image.Image, min_w: int, min_h: int, fill):
    w, h = img.size
    pad_w = max(0, min_w - w)
    pad_h = max(0, min_h - h)
    if pad_w == 0 and pad_h == 0:
        return img

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    return ImageOps.expand(img, border=(left, top, right, bottom), fill=fill)


class RandomPatchDataset(Dataset):
    """
    Training dataset: each item is a random crop patch from an image.

    - __len__ = num_images * patches_per_image
    - __getitem__ maps an index to an image, then samples a random patch.
    - Tries K times to find a patch containing crack pixels; falls back to random.
    """

    def __init__(
        self,
        image_dir,
        mask_dir,
        mask_prefix="auto",
        patch_size=512,
        patches_per_image=16,
        max_patch_tries=10,
        augment=False,
        image_transform=None,
        mask_transform=None,
        image_filenames=None,
        verbose=True,
        p_rotate=0.5,
        p_hflip=0.5,
        p_brightness=0.3,
        p_contrast=0.3,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_prefix = str(mask_prefix)
        self.patch_size = _normalize_patch_size(patch_size)
        self.patches_per_image = int(patches_per_image)
        self.max_patch_tries = int(max_patch_tries)
        self.augment = augment
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.verbose = verbose
        self.p_rotate = float(p_rotate)
        self.p_hflip = float(p_hflip)
        self.p_brightness = float(p_brightness)
        self.p_contrast = float(p_contrast)

        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

        if image_filenames is None:
            images = sorted(
                [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            )
            self.images = [
                img
                for img in images
                if find_mask_path(mask_dir, os.path.splitext(img)[0], mask_prefix=self.mask_prefix)
                is not None
            ]
        else:
            self.images = list(image_filenames)

        if not self.images:
            raise RuntimeError("No valid image-mask pairs found.")

        if self.verbose:
            print(f"RandomPatchDataset: {len(self.images)} image(s), patches_per_image={self.patches_per_image}")

    def __len__(self):
        return len(self.images) * self.patches_per_image

    def _load_pair(self, img_name):
        img_path = os.path.join(self.image_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        mask_path = find_mask_path(self.mask_dir, base_name, mask_prefix=self.mask_prefix)
        if mask_path is None:
            raise FileNotFoundError(
                f"Mask not found for '{img_name}'. Expected '{base_name}{self.mask_prefix}.*' in {self.mask_dir}"
            )

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        return image, mask

    def _random_patch(self, image: Image.Image, mask: Image.Image):
        patch_w, patch_h = self.patch_size
        image = _pad_to_min_size(image, patch_w, patch_h, fill=(0, 0, 0))
        mask = _pad_to_min_size(mask, patch_w, patch_h, fill=0)

        w, h = image.size
        left = 0 if w == patch_w else random.randint(0, w - patch_w)
        top = 0 if h == patch_h else random.randint(0, h - patch_h)
        box = (left, top, left + patch_w, top + patch_h)
        return image.crop(box), mask.crop(box)

    def __getitem__(self, idx):
        img_idx = int(idx) // self.patches_per_image
        img_idx = max(0, min(img_idx, len(self.images) - 1))
        img_name = self.images[img_idx]

        try:
            image, mask = self._load_pair(img_name)

            # Softer augmentation: each op has its own probability.
            if self.augment:
                if random.random() < self.p_rotate:
                    rot = random.choice([Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
                    image = image.transpose(rot)
                    mask = mask.transpose(rot)

                if random.random() < self.p_hflip:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

                if random.random() < self.p_brightness:
                    enhancer = ImageEnhance.Brightness(image)
                    image = enhancer.enhance(random.uniform(0.8, 1.2))

                if random.random() < self.p_contrast:
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(random.uniform(0.8, 1.2))

            best = None
            for _ in range(max(1, self.max_patch_tries)):
                img_p, mask_p = self._random_patch(image, mask)
                best = (img_p, mask_p)
                if mask_p.getbbox() is not None:
                    break

            image, mask = best

            if self.image_transform is not None:
                image = self.image_transform(image)
            else:
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
            else:
                mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0

            mask = (mask > 0.5).float()
            return image, mask
        except Exception as e:
            if self.verbose:
                print(f"RandomPatchDataset error at idx={idx} ({img_name}): {e}")
            # Skip bad samples (collate_fn should drop None).
            return None


class TiledDataset(Dataset):
    """
    Validation/Test dataset: deterministically tile each image into a full grid of patches.

    - stride = patch_size or patch_size//2 (overlap) to avoid cracks breaking on borders.
    - No random geometric augmentation.
    """

    def __init__(
        self,
        image_dir,
        mask_dir,
        mask_prefix="auto",
        patch_size=512,
        stride=None,
        image_transform=None,
        mask_transform=None,
        image_filenames=None,
        verbose=True,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_prefix = str(mask_prefix)
        self.patch_size = _normalize_patch_size(patch_size)
        self.stride = _normalize_patch_size(stride) if stride is not None else None
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.verbose = verbose

        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

        if image_filenames is None:
            images = sorted(
                [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            )
            self.images = [
                img
                for img in images
                if find_mask_path(mask_dir, os.path.splitext(img)[0], mask_prefix=self.mask_prefix)
                is not None
            ]
        else:
            self.images = list(image_filenames)

        if not self.images:
            raise RuntimeError("No valid image-mask pairs found.")

        patch_w, patch_h = self.patch_size
        if self.stride is None:
            self.stride = (patch_w // 2, patch_h // 2)
        stride_w, stride_h = self.stride
        if stride_w <= 0 or stride_h <= 0:
            raise ValueError("stride must be > 0")

        self.index = []
        for img_name in self.images:
            img_path = os.path.join(self.image_dir, img_name)
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception as e:
                if self.verbose:
                    print(f"TiledDataset: skipping unreadable image '{img_name}': {e}")
                continue

            w = max(w, patch_w)
            h = max(h, patch_h)

            xs = list(range(0, max(1, w - patch_w + 1), stride_w))
            ys = list(range(0, max(1, h - patch_h + 1), stride_h))
            if xs[-1] != w - patch_w:
                xs.append(w - patch_w)
            if ys[-1] != h - patch_h:
                ys.append(h - patch_h)

            for y in ys:
                for x in xs:
                    self.index.append((img_name, int(x), int(y)))

        if self.verbose:
            print(
                f"TiledDataset: {len(self.images)} image(s) -> {len(self.index)} patch(es), "
                f"patch_size={self.patch_size}, stride={self.stride}"
            )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        img_name, x, y = self.index[int(idx)]
        try:
            img_path = os.path.join(self.image_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            mask_path = find_mask_path(self.mask_dir, base_name, mask_prefix=self.mask_prefix)
            if mask_path is None:
                raise FileNotFoundError(
                    f"Mask not found for '{img_name}'. Expected '{base_name}{self.mask_prefix}.*' in {self.mask_dir}"
                )

            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            patch_w, patch_h = self.patch_size
            image = _pad_to_min_size(image, patch_w, patch_h, fill=(0, 0, 0))
            mask = _pad_to_min_size(mask, patch_w, patch_h, fill=0)

            box = (x, y, x + patch_w, y + patch_h)
            image = image.crop(box)
            mask = mask.crop(box)

            if self.image_transform is not None:
                image = self.image_transform(image)
            else:
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
            else:
                mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0

            mask = (mask > 0.5).float()
            return image, mask
        except Exception as e:
            if self.verbose:
                print(f"TiledDataset error at idx={idx} ({img_name}): {e}")
            # Skip bad samples (collate_fn should drop None).
            return None


class CrackDataset(Dataset):
    """
    Crack dataset loader.

    Responsibilities:
    - Loads images and their corresponding masks.
    - Optionally applies data augmentation to improve generalization.
    - Returns tensors in a consistent, model-ready format.

    Args:
        image_dir: Directory containing input images.
        mask_dir: Directory containing mask images.
        transform: Optional transform applied to both image and mask (legacy).
        image_transform: Transform applied only to the image.
        mask_transform: Transform applied only to the mask.
        augment: Whether to enable random data augmentation.
        patch_size: If set, return a cropped patch (e.g., 256) from the original image.
        patch_strategy: "random" (optionally biased to crack pixels) or "center".
        max_patch_tries: Number of random tries to find a patch containing cracks.
        output_size: Used for safe fallback tensors if a sample fails to load.
        image_filenames: Optional list of image filenames to use (pre-split train/val).
        verbose: Print dataset summary.
    """
    def __init__(
        self,
        image_dir,
        mask_dir,
        transform=None,
        image_transform=None,
        mask_transform=None,
        mask_prefix="auto",
        augment=False,
        patch_size=None,
        patch_strategy="random",
        max_patch_tries=10,
        output_size=256,
        image_filenames=None,
        verbose=True,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.mask_prefix = str(mask_prefix)
        self.augment = augment
        self.patch_size = patch_size
        self.patch_strategy = patch_strategy
        self.max_patch_tries = max_patch_tries
        self.output_size = int(output_size)
        self.verbose = verbose

        if self.image_transform is None and self.mask_transform is None and self.transform is not None:
            # Backwards compatibility: previous code passed a single transform for both.
            self.image_transform = self.transform
            self.mask_transform = self.transform
        
        # Check if directories exist
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")
            
        self.patch_size = self._normalize_patch_size(self.patch_size)

        # Collect image filenames (only image extensions), or use the provided list.
        if image_filenames is None:
            self.images = sorted(
                [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            )
        else:
            self.images = list(image_filenames)
        
        # Validate pairs: keep only images with a matching mask.
        valid_images = []
        for img in self.images:
            base_name = os.path.splitext(img)[0]
            mask_path = find_mask_path(mask_dir, base_name, mask_prefix=self.mask_prefix)
            if mask_path is not None:
                valid_images.append(img)
            
        self.images = valid_images
        if self.verbose:
            print(f"Found {len(self.images)} valid image-mask pairs")
        
        # Print the first 2 samples to help debugging.
        if self.verbose:
            for i in range(min(2, len(self.images))):
                img_name = self.images[i]
                base_name = os.path.splitext(img_name)[0]
                mask_path = find_mask_path(mask_dir, base_name, mask_prefix=self.mask_prefix)
                print(f"Image {i}: {img_name}")
                print(f"Mask  {i}: {os.path.basename(mask_path) if mask_path else '(missing)'}")

    @staticmethod
    def list_valid_images(image_dir, mask_dir, mask_prefix="auto"):
        return _list_valid_images(image_dir, mask_dir, mask_prefix=mask_prefix)

    @staticmethod
    def _normalize_patch_size(patch_size):
        return _normalize_patch_size(patch_size)

    @staticmethod
    def _pad_to_min_size(img: Image.Image, min_w: int, min_h: int, fill):
        return _pad_to_min_size(img, min_w, min_h, fill)

    def _crop_patch(self, image: Image.Image, mask: Image.Image):
        patch_w, patch_h = self.patch_size

        image = self._pad_to_min_size(image, patch_w, patch_h, fill=(0, 0, 0))
        mask = self._pad_to_min_size(mask, patch_w, patch_h, fill=0)

        w, h = image.size
        if self.patch_strategy == "center":
            left = max(0, (w - patch_w) // 2)
            top = max(0, (h - patch_h) // 2)
            box = (left, top, left + patch_w, top + patch_h)
            return image.crop(box), mask.crop(box)

        # Random crop, optionally biased toward patches that contain cracks.
        best = None
        for _ in range(max(1, int(self.max_patch_tries))):
            left = 0 if w == patch_w else random.randint(0, w - patch_w)
            top = 0 if h == patch_h else random.randint(0, h - patch_h)
            box = (left, top, left + patch_w, top + patch_h)
            img_patch = image.crop(box)
            mask_patch = mask.crop(box)
            best = (img_patch, mask_patch)

            # Fast check: if this patch has any non-zero pixels, it likely contains crack.
            if mask_patch.getbbox() is not None:
                return img_patch, mask_patch

        return best
        
    def __len__(self):
        """Return the size of the dataset."""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Return one sample (image, mask), optionally applying augmentation.

        Notes:
        - Image and mask must receive the same geometric transforms to stay aligned.
        - Brightness/contrast should only be applied to the image (not the mask).
        - Exceptions are handled so a single bad sample doesn't stop training.
        """
        try:
            # Load image
            img_name = self.images[idx]
            img_path = os.path.join(self.image_dir, img_name)
            
            # Build mask path using the unified naming rule: {base}{mask_prefix}.*
            base_name = os.path.splitext(img_name)[0]
            mask_path = find_mask_path(self.mask_dir, base_name, mask_prefix=self.mask_prefix)
            if mask_path is None:
                raise FileNotFoundError(
                    f"Mask not found for '{img_name}'. Expected '{base_name}{self.mask_prefix}.*' in {self.mask_dir}"
                )
            
            # Read images and masks, and convert to appropriate formats.
            image = Image.open(img_path).convert('RGB')  # Ensure 3 channels
            mask = Image.open(mask_path).convert('L')  # Single-channel grayscale
            
            # Data augmentation (randomly applied during training)
            if self.augment:
                if random.random() < 0.5:
                    rot = random.choice([Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
                    image = image.transpose(rot)
                    mask = mask.transpose(rot)

                if random.random() < 0.5:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

                if random.random() < 0.3:
                    enhancer = ImageEnhance.Brightness(image)
                    image = enhancer.enhance(random.uniform(0.8, 1.2))

                if random.random() < 0.3:
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(random.uniform(0.8, 1.2))

            # Patch-based training: crop a 256x256 (or configured) patch at original resolution.
            if self.patch_size is not None:
                image, mask = self._crop_patch(image, mask)
            
            # Apply additional transforms (resize, convert to tensor, etc.)
            if self.image_transform is not None:
                image = self.image_transform(image)
            else:
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
            else:
                mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0
            
            # Ensure the mask is binary (0 or 1).
            mask = (mask > 0.5).float()  # Pixels > 0.5 are cracks (1), else background (0)
            
            return image, mask
            
        except Exception as e:
            # Error handling: prevent a single sample error from stopping the entire training.
            print(f"Error processing image at index {idx}: {e}")
            # Skip bad samples (collate_fn should drop None).
            return None

class AttentionGate(nn.Module):
    """
    Attention Gate module.

    This gate learns attention coefficients to suppress irrelevant background features
    and highlight crack-related regions in skip connections.

    Args:
        F_g: Number of channels for upsampling features.
        F_l: Number of channels for skip-connection features.
        F_int: Number of channels for intermediate features (dimensionality reduction).
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        # Convolution layer for processing upsampling features
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),  # 1x1 conv for channel reduction
            nn.BatchNorm2d(F_int),
        )
        # Convolution layer for processing skip-connection features
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),  # 1x1 conv for channel reduction
            nn.BatchNorm2d(F_int),
        )
        # Convolution layer for generating the attention map
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),  # Output a single-channel attention map
            nn.BatchNorm2d(1),
            nn.Sigmoid(),  # Limit values to [0, 1]
        )
        # Activation function
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        """
        Compute attention coefficients and apply them to the skip features.

        Args:
            g: Upsampled features from the decoder.
            x: Skip-connection features from the encoder.

        Returns:
            The reweighted skip-connection features.
        """
        # Dimensionality reduction
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Feature fusion and activation
        psi = self.relu(g1 + x1)
        
        # Generate attention coefficients in [0, 1]
        psi = self.psi(psi)
        
        # Apply attention to the original skip features
        return x * psi
