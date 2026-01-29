import torch


def dice_score(pred_logits, target, thr=0.5, eps=1e-6):
    pred = (torch.sigmoid(pred_logits) > thr).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()


def iou_score(pred_logits, target, thr=0.5, eps=1e-6):
    pred = (torch.sigmoid(pred_logits) > thr).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()


def dice_score_from_prob(pred_prob, target, thr=0.5, eps=1e-6):
    pred = (pred_prob > thr).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()


def iou_score_from_prob(pred_prob, target, thr=0.5, eps=1e-6):
    pred = (pred_prob > thr).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()
