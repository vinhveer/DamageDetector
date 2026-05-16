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


def _skeletonize_mask(mask_np: np.ndarray) -> np.ndarray:
    work = (mask_np > 0).astype(np.uint8)
    skeleton = np.zeros_like(work, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(work) > 0:
        eroded = cv2.erode(work, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
        work = eroded
    return skeleton


def sample_positive_points(mask_np: np.ndarray, *, split: str, max_points: int = 8) -> tuple[np.ndarray, np.ndarray]:
    h_img, w_img = mask_np.shape
    point_coords = np.zeros((max_points, 2), dtype=np.float32)
    point_labels = -np.ones((max_points,), dtype=np.float32)
    if int(mask_np.sum()) <= 0:
        return point_coords, point_labels

    skeleton_coords = np.argwhere(_skeletonize_mask(mask_np) > 0)
    coords = skeleton_coords if len(skeleton_coords) > 0 else np.argwhere(mask_np > 0)
    if len(coords) == 0:
        return point_coords, point_labels

    if split == "train":
        n_points = int(np.random.randint(5, max_points + 1))
        replace = len(coords) < n_points
        indices = np.random.choice(len(coords), size=n_points, replace=replace)
    else:
        n_points = min(max_points, len(coords))
        indices = np.linspace(0, len(coords) - 1, num=n_points, dtype=np.int64)

    selected = coords[indices][:max_points]
    point_coords[: len(selected)] = selected[:, ::-1].astype(np.float32)
    point_coords[:, 0] = np.clip(point_coords[:, 0], 0, w_img - 1)
    point_coords[:, 1] = np.clip(point_coords[:, 1], 0, h_img - 1)
    point_labels[: len(selected)] = 1.0
    return point_coords, point_labels


def sample_negative_points(mask_np: np.ndarray, *, split: str, max_points: int = 2) -> tuple[np.ndarray, np.ndarray]:
    point_coords = np.zeros((max_points, 2), dtype=np.float32)
    point_labels = -np.ones((max_points,), dtype=np.float32)
    for index in range(max_points):
        point_coords[index] = sample_negative_point(mask_np, split=split)
        point_labels[index] = 0.0
    return point_coords, point_labels


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
    recipe_pos_coords, recipe_pos_labels = sample_positive_points(mask_np, split=split, max_points=8)
    recipe_neg_coords, recipe_neg_labels = sample_negative_points(mask_np, split=split, max_points=2)
    recipe_point_coords = np.concatenate([recipe_pos_coords, recipe_neg_coords], axis=0)
    recipe_point_labels = np.concatenate([recipe_pos_labels, recipe_neg_labels], axis=0)
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
        "recipe_point_coords": torch.from_numpy(recipe_point_coords),
        "recipe_point_labels": torch.from_numpy(recipe_point_labels),
        "case_name": case_name,
    }
