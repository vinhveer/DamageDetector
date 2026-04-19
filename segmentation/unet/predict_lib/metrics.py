import numpy as np
from PIL import Image


def _load_binary_mask(mask_path: str, target_size=None) -> np.ndarray:
    mask_img = Image.open(mask_path).convert("L")
    if target_size is not None:
        mask_img = mask_img.resize(target_size, resample=Image.NEAREST)
    mask = np.array(mask_img, dtype=np.uint8) > 127
    return mask


def _dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-8) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum(dtype=np.float64)
    denom = pred.sum(dtype=np.float64) + gt.sum(dtype=np.float64)
    if denom == 0:
        return 1.0
    return float((2.0 * inter + eps) / (denom + eps))


def load_binary_mask(mask_path: str, target_size=None) -> np.ndarray:
    return _load_binary_mask(mask_path, target_size=target_size)


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-8) -> float:
    return _dice_score(pred_mask, gt_mask, eps=eps)


def evaluate_against_gt(binary_mask: np.ndarray, gt_mask_path: str | None, target_size):
    if not gt_mask_path:
        return None, None
    try:
        gt_mask = _load_binary_mask(gt_mask_path, target_size=target_size)
        dice = _dice_score(binary_mask, gt_mask)
        return gt_mask, dice
    except FileNotFoundError:
        return None, None
    except Exception:
        return None, None


def mask_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> dict:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, np.logical_not(gt_b)).sum())
    fn = int(np.logical_and(np.logical_not(pred_b), gt_b).sum())
    tn = int(np.logical_and(np.logical_not(pred_b), np.logical_not(gt_b)).sum())
    total = tp + fp + fn + tn

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    accuracy = (tp + tn + eps) / (total + eps)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
    }


def diff_rgb(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    h, w = pred_b.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    tp = np.logical_and(pred_b, gt_b)
    fp = np.logical_and(pred_b, np.logical_not(gt_b))
    fn = np.logical_and(np.logical_not(pred_b), gt_b)

    rgb[tp] = (0, 255, 0)     # TP: green
    rgb[fp] = (255, 0, 0)     # FP: red
    rgb[fn] = (0, 0, 255)     # FN: blue
    return rgb
