from torch_runtime import F, torch


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


def soft_skeleton(prob, num_iter=10):
    """Differentiable skeleton approximation for thin-structure supervision."""
    img = prob.clamp(0.0, 1.0)
    skel = F.relu(img - F.max_pool2d(-img, kernel_size=3, stride=1, padding=1).neg())
    for _ in range(int(num_iter)):
        img = F.max_pool2d(-img, kernel_size=3, stride=1, padding=1).neg()
        delta = F.relu(img - F.max_pool2d(-img, kernel_size=3, stride=1, padding=1).neg())
        skel = skel + F.relu(delta - skel * delta)
    return skel.clamp(0.0, 1.0)


def cl_dice_loss(pred_logits, target, smooth=1.0, num_iter=10):
    prob = torch.sigmoid(pred_logits)
    target = target.float()
    s_pred = soft_skeleton(prob, num_iter=num_iter)
    s_gt = soft_skeleton(target, num_iter=num_iter)
    t_prec = (s_pred * target).sum(dim=(1, 2, 3)) / (s_pred.sum(dim=(1, 2, 3)) + smooth)
    t_rec = (s_gt * prob).sum(dim=(1, 2, 3)) / (s_gt.sum(dim=(1, 2, 3)) + smooth)
    cl_dice = (2.0 * t_prec * t_rec + smooth) / (t_prec + t_rec + smooth)
    return 1.0 - cl_dice.mean()


def centerline_target_from_mask(target, num_iter=10):
    return soft_skeleton(target.float(), num_iter=num_iter).detach()
