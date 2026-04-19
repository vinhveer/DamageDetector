import cv2
import numpy as np
from torch_runtime import torch


def ensure_numpy_image_label(image, label):
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    if isinstance(label, torch.Tensor):
        label = label.numpy()
    return image, label


def ensure_three_channel_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image[:, :, None]
    return image


def normalize_binary_mask(label: np.ndarray) -> np.ndarray:
    if label.max() > 1:
        label = label / 255.0
    return label


def pad_canvas_if_needed(
    image: np.ndarray,
    label: np.ndarray,
    *,
    min_h: int,
    min_w: int,
    image_border_mode: int = cv2.BORDER_CONSTANT,
    image_fill=0,
    label_border_mode: int = cv2.BORDER_CONSTANT,
    label_fill=0,
):
    h, w = label.shape[:2]
    if w >= min_w and h >= min_h:
        return image, label

    pad_w = max(0, min_w - w)
    pad_h = max(0, min_h - h)
    padded_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, image_border_mode, value=image_fill)
    padded_label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, label_border_mode, value=label_fill)
    return padded_image, padded_label


def crop_image_label(image: np.ndarray, label: np.ndarray, *, y1: int, x1: int, th: int, tw: int):
    cropped_image = image[y1:y1 + th, x1:x1 + tw]
    cropped_label = label[y1:y1 + th, x1:x1 + tw]
    return cropped_image, cropped_label


def to_uint8_image(image: np.ndarray) -> np.ndarray:
    if image.dtype in (np.float32, np.float64):
        if image.max() <= 1.0:
            return (image * 255.0).astype(np.uint8)
        return image.astype(np.uint8)
    return image.astype(np.uint8)


def resize_with_center_padding(
    image: np.ndarray,
    label: np.ndarray,
    *,
    target_h: int,
    target_w: int,
    normalize_image: bool = True,
):
    if image.ndim == 3:
        src_h, src_w, channels = image.shape
    else:
        src_h, src_w = image.shape
        channels = 1

    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    if new_w != src_w or new_h != src_h:
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if channels == 1 and image.ndim == 2:
            image = image[:, :, None]
        label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    final_image = np.zeros((target_h, target_w, channels), dtype=np.float32)
    final_label = np.zeros((target_h, target_w), dtype=np.float32)

    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    if channels > 1:
        final_image[pad_top:pad_top + new_h, pad_left:pad_left + new_w, :] = image
    else:
        final_image[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = image
    final_label[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = label

    if normalize_image:
        final_image = np.clip(final_image / 255.0, 0, 1)
    return final_image, final_label
