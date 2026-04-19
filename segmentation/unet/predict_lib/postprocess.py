import numpy as np
from PIL import Image, ImageFilter


def binarize_prediction(pred: np.ndarray, threshold: float) -> np.ndarray:
    return pred > threshold


def postprocess_binary_mask(binary_mask: np.ndarray, apply_postprocessing: bool = True) -> np.ndarray:
    if not apply_postprocessing:
        return binary_mask
    mask_img = Image.fromarray(binary_mask.astype(np.uint8) * 255).convert("L")
    mask_img = mask_img.filter(ImageFilter.MedianFilter(size=3))
    return np.array(mask_img) > 127


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
