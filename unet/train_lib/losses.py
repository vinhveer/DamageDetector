import torch
import torch.nn.functional as F


def dice_loss(pred, target):
    smooth = 1.0
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def focal_loss_with_logits(pred, target, alpha=0.25, gamma=2.0, reduction="mean"):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    prob = torch.sigmoid(pred)
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    if alpha is None:
        alpha_t = 1.0
    else:
        alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = alpha_t * (1.0 - p_t) ** gamma * bce
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    return loss.mean()
