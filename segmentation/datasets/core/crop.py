import random

import cv2
import numpy as np


def build_crop_metadata(
    image: np.ndarray,
    label: np.ndarray,
    *,
    crop_policy: str = "smart",
) -> dict[str, np.ndarray]:
    crop_policy = str(crop_policy or "smart").strip().lower()
    mask = (label > 0).astype(np.uint8)
    metadata: dict[str, np.ndarray] = {
        "foreground_coords": np.argwhere(mask > 0).astype(np.int32),
    }
    if crop_policy == "fast":
        return metadata

    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY) if image.ndim == 3 and image.shape[2] == 3 else image.astype(np.uint8)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    metadata["edge_map"] = cv2.magnitude(sobel_x, sobel_y)

    if int(mask.sum()) > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, kernel, iterations=1)
        boundary = np.logical_and(dilated > 0, eroded == 0)
        ambiguous_bg = np.logical_and(dilated > 0, mask == 0)
        metadata["boundary_coords"] = np.argwhere(boundary).astype(np.int32)
        metadata["ambiguous_bg_coords"] = np.argwhere(ambiguous_bg).astype(np.int32)
    else:
        metadata["boundary_coords"] = np.empty((0, 2), dtype=np.int32)
        metadata["ambiguous_bg_coords"] = np.empty((0, 2), dtype=np.int32)
    return metadata


def random_crop_coords(h: int, w: int, th: int, tw: int) -> tuple[int, int]:
    y1 = random.randint(0, max(0, h - th))
    x1 = random.randint(0, max(0, w - tw))
    return int(y1), int(x1)


def hard_negative_crop_coords(
    image: np.ndarray,
    label: np.ndarray,
    h: int,
    w: int,
    th: int,
    tw: int,
    *,
    num_trials: int = 32,
    edge_map: np.ndarray | None = None,
) -> tuple[int, int]:
    if edge_map is None:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY) if image.ndim == 3 and image.shape[2] == 3 else image.astype(np.uint8)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_map = cv2.magnitude(sobel_x, sobel_y)

    best_score = -1.0
    best_coords = random_crop_coords(h, w, th, tw)
    for _ in range(max(1, int(num_trials))):
        y1, x1 = random_crop_coords(h, w, th, tw)
        label_crop = label[y1:y1 + th, x1:x1 + tw]
        if int((label_crop > 0).sum()) > 0:
            continue
        edge_crop = edge_map[y1:y1 + th, x1:x1 + tw]
        score = float(edge_crop.mean())
        if score > best_score:
            best_score = score
            best_coords = (y1, x1)
    return int(best_coords[0]), int(best_coords[1])


def choose_smart_crack_crop_coords(
    image: np.ndarray,
    label: np.ndarray,
    h: int,
    w: int,
    th: int,
    tw: int,
    *,
    background_crop_prob: float = 0.2,
    near_background_crop_prob: float = 0.15,
    hard_negative_crop_prob: float = 0.10,
    crop_policy: str = "smart",
    metadata: dict[str, np.ndarray] | None = None,
) -> tuple[int, int]:
    foreground_coords = metadata.get("foreground_coords") if metadata is not None else None
    if foreground_coords is not None:
        y_inds = foreground_coords[:, 0]
        x_inds = foreground_coords[:, 1]
    else:
        y_inds, x_inds = np.where(label > 0)
    crop_policy = str(crop_policy or "smart").strip().lower()
    background_crop_prob = float(background_crop_prob)
    near_background_crop_prob = float(near_background_crop_prob)
    hard_negative_crop_prob = float(hard_negative_crop_prob)

    if crop_policy == "fast":
        if len(y_inds) > 0:
            crop_mode = random.random()
            if crop_mode < background_crop_prob:
                return random_crop_coords(h, w, th, tw)
            idx = random.randint(0, len(y_inds) - 1)
            cy, cx = int(y_inds[idx]), int(x_inds[idx])
            offset_scale = 0.75 if crop_mode < background_crop_prob + near_background_crop_prob else 0.35
            cy += random.randint(-int(th * offset_scale), int(th * offset_scale))
            cx += random.randint(-int(tw * offset_scale), int(tw * offset_scale))
            y1 = max(0, min(h - th, cy - th // 2))
            x1 = max(0, min(w - tw, cx - tw // 2))
            return int(y1), int(x1)
        return random_crop_coords(h, w, th, tw)

    if len(y_inds) > 0:
        crop_mode = random.random()
        if crop_mode < background_crop_prob:
            return random_crop_coords(h, w, th, tw)
        if crop_mode < background_crop_prob + hard_negative_crop_prob:
            return hard_negative_crop_coords(
                image,
                label,
                h,
                w,
                th,
                tw,
                edge_map=metadata.get("edge_map") if metadata is not None else None,
            )

        idx = random.randint(0, len(y_inds) - 1)
        cy, cx = int(y_inds[idx]), int(x_inds[idx])
        if crop_mode < background_crop_prob + hard_negative_crop_prob + near_background_crop_prob:
            offset_y = random.randint(-int(th * 0.9), int(th * 0.9))
            offset_x = random.randint(-int(tw * 0.9), int(tw * 0.9))
        else:
            offset_y = random.randint(-int(th * 0.4), int(th * 0.4))
            offset_x = random.randint(-int(tw * 0.4), int(tw * 0.4))
        cy += offset_y
        cx += offset_x
        y1 = max(0, min(h - th, cy - th // 2))
        x1 = max(0, min(w - tw, cx - tw // 2))
        return int(y1), int(x1)

    if random.random() < hard_negative_crop_prob:
        return hard_negative_crop_coords(
            image,
            label,
            h,
            w,
            th,
            tw,
            edge_map=metadata.get("edge_map") if metadata is not None else None,
        )
    return random_crop_coords(h, w, th, tw)


def choose_refine_crop_coords(
    image: np.ndarray,
    label: np.ndarray,
    h: int,
    w: int,
    th: int,
    tw: int,
    *,
    crop_policy: str = "smart",
    metadata: dict[str, np.ndarray] | None = None,
) -> tuple[int, int]:
    mask = (label > 0).astype(np.uint8)
    coords = metadata.get("foreground_coords") if metadata is not None else np.argwhere(mask > 0)
    if len(coords) == 0:
        return choose_smart_crack_crop_coords(image, label, h, w, th, tw, crop_policy=crop_policy, metadata=metadata)

    crop_policy = str(crop_policy or "smart").strip().lower()
    if crop_policy == "fast":
        point = coords[np.random.randint(len(coords))]
        cy, cx = int(point[0]), int(point[1])
        cy += random.randint(-int(th * 0.25), int(th * 0.25))
        cx += random.randint(-int(tw * 0.25), int(tw * 0.25))
        y1 = max(0, min(h - th, cy - th // 2))
        x1 = max(0, min(w - tw, cx - tw // 2))
        return int(y1), int(x1)

    if metadata is not None:
        boundary_coords = metadata.get("boundary_coords", np.empty((0, 2), dtype=np.int32))
        ambiguous_bg_coords = metadata.get("ambiguous_bg_coords", np.empty((0, 2), dtype=np.int32))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, kernel, iterations=1)
        boundary = np.logical_and(dilated > 0, eroded == 0)
        ambiguous_bg = np.logical_and(dilated > 0, mask == 0)
        boundary_coords = np.argwhere(boundary)
        ambiguous_bg_coords = np.argwhere(ambiguous_bg)

    mode = random.random()
    if mode < 0.45:
        selected = coords
        offset_scale = 0.20
    elif mode < 0.75 and len(boundary_coords) > 0:
        selected = boundary_coords
        offset_scale = 0.15
    elif len(ambiguous_bg_coords) > 0:
        selected = ambiguous_bg_coords
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


def choose_centered_foreground_crop_coords(
    label: np.ndarray,
    h: int,
    w: int,
    th: int,
    tw: int,
    *,
    metadata: dict[str, np.ndarray] | None = None,
) -> tuple[int, int]:
    foreground_coords = metadata.get("foreground_coords") if metadata is not None else None
    if foreground_coords is not None:
        y_inds = foreground_coords[:, 0]
        x_inds = foreground_coords[:, 1]
    else:
        y_inds, x_inds = np.where(label > 0)
    if len(y_inds) > 0:
        cy = int(np.median(y_inds))
        cx = int(np.median(x_inds))
        y1 = max(0, min(cy - th // 2, h - th))
        x1 = max(0, min(cx - tw // 2, w - tw))
    else:
        y1 = max(0, (h - th) // 2)
        x1 = max(0, (w - tw) // 2)
    return int(y1), int(x1)
