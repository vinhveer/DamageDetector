import numpy as np
import cv2
from torch_runtime import torch


def random_rot_flip(image, label):
    # Deprecated: handled by Albumentations.
    pass


def random_rotate(image, label):
    # Deprecated: handled by Albumentations.
    pass


def select_point(coords: np.ndarray, *, split: str) -> np.ndarray:
    if len(coords) == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    if split == "train":
        point = coords[np.random.randint(len(coords))]
    else:
        point = coords[len(coords) // 2]
    return point[::-1].astype(np.float32)


def sample_negative_point(mask_np: np.ndarray, *, split: str) -> np.ndarray:
    ring_coords = np.empty((0, 2), dtype=np.int32)
    if int(mask_np.sum()) > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated = cv2.dilate(mask_np.astype(np.uint8), kernel, iterations=1)
        ring = np.logical_and(dilated > 0, mask_np == 0)
        ring_coords = np.argwhere(ring)
    if len(ring_coords) > 0:
        return select_point(ring_coords, split=split)

    bg_coords = np.argwhere(mask_np == 0)
    if len(bg_coords) > 0:
        return select_point(bg_coords, split=split)

    h_img, w_img = mask_np.shape
    return np.array([float(w_img // 2), float(h_img // 2)], dtype=np.float32)


def build_prompt_tensors(mask_np: np.ndarray, *, split: str, use_full_image_box: bool, case_name: str) -> dict:
    coords = np.argwhere(mask_np > 0)
    h, w = mask_np.shape
    has_foreground = len(coords) > 0
    full_box = np.array([0, 0, w, h], dtype=np.float32)

    if has_foreground:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        if split == "train":
            pad_x1 = np.random.randint(0, 10)
            pad_y1 = np.random.randint(0, 10)
            pad_x2 = np.random.randint(0, 10)
            pad_y2 = np.random.randint(0, 10)
        else:
            pad_x1 = pad_y1 = pad_x2 = pad_y2 = 5

        x_min = max(0, x_min - pad_x1)
        y_min = max(0, y_min - pad_y1)
        x_max = min(w, x_max + pad_x2)
        y_max = min(h, y_max + pad_y2)
        tight_box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
        pos_pt = select_point(coords, split=split)
    else:
        tight_box = full_box.copy()
        pos_pt = np.array([float(w // 2), float(h // 2)], dtype=np.float32)

    neg_pt = sample_negative_point(mask_np, split=split)
    box = full_box if use_full_image_box or not has_foreground else tight_box
    if has_foreground:
        point_coords = np.array([pos_pt, neg_pt], dtype=np.float32)
        point_labels = np.array([1, 0], dtype=np.float32)
    else:
        point_coords = np.array([neg_pt, neg_pt], dtype=np.float32)
        point_labels = np.array([0, 0], dtype=np.float32)

    return {
        "box": torch.from_numpy(box),
        "full_box": torch.from_numpy(full_box),
        "tight_box": torch.from_numpy(tight_box),
        "pos_point": torch.from_numpy(pos_pt),
        "neg_point": torch.from_numpy(neg_pt),
        "has_foreground": torch.tensor(bool(has_foreground)),
        "point_coords": torch.from_numpy(point_coords),
        "point_labels": torch.from_numpy(point_labels),
        "case_name": case_name,
    }
