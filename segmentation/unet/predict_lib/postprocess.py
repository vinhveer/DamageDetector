import numpy as np
import cv2


def binarize_prediction(pred: np.ndarray, threshold: float) -> np.ndarray:
    return pred > threshold


def postprocess_binary_mask(
    binary_mask: np.ndarray,
    apply_postprocessing: bool = True,
    min_size: int = 50,
) -> np.ndarray:
    if not apply_postprocessing:
        return binary_mask.astype(bool)
    min_size = int(min_size)
    mask = binary_mask.astype(bool)
    if min_size <= 0 or not mask.any():
        return mask

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    cleaned = np.zeros(mask.shape, dtype=bool)
    for label_idx in range(1, num_labels):
        if int(stats[label_idx, cv2.CC_STAT_AREA]) >= min_size:
            cleaned[labels == label_idx] = True
    return cleaned


def restore_prediction_to_full(
    pred: np.ndarray,
    binary_mask: np.ndarray,
    roi_box,
    full_size,
):
    if roi_box is None:
        return pred, binary_mask

    full_w, full_h = full_size
    left, top, right, bottom = roi_box

    pred_full = np.zeros((full_h, full_w), dtype=np.float32)
    pred_full[top:bottom, left:right] = pred.astype(np.float32)

    binary_full = np.zeros((full_h, full_w), dtype=bool)
    binary_full[top:bottom, left:right] = binary_mask.astype(bool)

    return pred_full, binary_full
