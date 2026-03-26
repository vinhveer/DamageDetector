import cv2
import numpy as np

def compute_dice_iou(mask1: np.ndarray, mask2: np.ndarray) -> tuple[float, float]:
    """
    Computes Dice coefficient and IoU for two binary masks.
    Input masks should be numpy arrays.
    Returns (dice, iou)
    """
    # Ensure binary
    m1 = (mask1 > 0).astype(bool)
    m2 = (mask2 > 0).astype(bool)

    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()

    if union == 0:
        return 1.0, 1.0 # Both empty -> perfect match

    iou = intersection / union
    dice = 2.0 * intersection / (m1.sum() + m2.sum())

    return float(dice), float(iou)
