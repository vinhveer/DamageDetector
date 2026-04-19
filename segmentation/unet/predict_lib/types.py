from dataclasses import dataclass

import numpy as np
from PIL import Image


class StopRequested(RuntimeError):
    pass


@dataclass
class PredictionResult:
    image_path: str
    image: Image.Image
    pred: np.ndarray
    binary_mask: np.ndarray
    roi_box: tuple[int, int, int, int] | None
    dice: float | None
    gt_mask: np.ndarray | None
    gt_mask_path: str | None
    original_size: tuple[int, int]
