import os

import numpy as np
from PIL import Image

from .types import PredictionResult
from .visualize import build_preview_arrays, save_prediction_preview


def save_binary_mask(mask_path: str, binary_mask: np.ndarray, output_size=None):
    binary_mask_out = binary_mask.astype(np.uint8) * 255
    mask_output = Image.fromarray(binary_mask_out)
    if output_size is not None:
        mask_output = mask_output.resize(output_size, resample=Image.NEAREST)
    mask_output.save(mask_path)


def save_prediction_outputs(
    result: PredictionResult,
    output_dir: str,
    base_name: str,
    *,
    threshold: float,
    gt_expected: bool = False,
    max_side: int = 1024,
):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}_prediction.png")
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")

    img_v, pred_v, binary_v = build_preview_arrays(result.image, result.pred, result.binary_mask, max_side=max_side)
    show_gt_panel = bool(gt_expected) or (result.gt_mask is not None)
    save_prediction_preview(
        output_path,
        img_v,
        pred_v,
        binary_v,
        threshold=threshold,
        dice=result.dice,
        gt_mask=result.gt_mask,
        show_gt_panel=show_gt_panel,
    )

    save_binary_mask(mask_path, result.binary_mask, result.original_size)
    return output_path, mask_path
