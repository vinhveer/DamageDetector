import numpy as np
from torch_runtime import torch
from scipy.ndimage import zoom
from torch_runtime import nn
from torch_runtime import F
from PIL import Image
import cv2

class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        Calc focal loss
        :param preds: size: [B, N, C] or [B, C], corresponds to detection and classification tasks  [B, C, H, W]: segmentation
        :param labels: size: [B, N] or [B]  [B, H, W]: segmentation
        :return:
        """
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.low(1 - preds_softmax) == (1 - pt) ** r

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1.0
        intersect = torch.sum(score * target) 
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        target = target.float()
        if target.ndim == 3:
            target = target.unsqueeze(1)

        probs = probs.reshape(probs.shape[0], -1)
        target = target.reshape(target.shape[0], -1)

        intersect = (probs * target).sum(dim=1)
        denom = probs.sum(dim=1) + target.sum(dim=1)
        dice = (2 * intersect + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class BinaryTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        target = target.float()
        if target.ndim == 3:
            target = target.unsqueeze(1)

        probs = probs.reshape(probs.shape[0], -1)
        target = target.reshape(target.shape[0], -1)

        tp = (probs * target).sum(dim=1)
        fp = (probs * (1.0 - target)).sum(dim=1)
        fn = ((1.0 - probs) * target).sum(dim=1)
        score = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - score.mean()


class BinaryFocalWithLogitsLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = str(reduction)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        if target.ndim == 3:
            target = target.unsqueeze(1)

        probs = torch.sigmoid(logits)
        pt = probs * target + (1.0 - probs) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal_weight = alpha_t * torch.pow(torch.clamp(1.0 - pt, min=0.0), self.gamma)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = focal_weight * bce

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    if img.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {tuple(img.shape)}")
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    if img.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {tuple(img.shape)}")
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    img = torch.clamp(img, 0.0, 1.0)
    skeleton = F.relu(img - _soft_open(img))
    work = img
    for _ in range(max(0, int(iterations) - 1)):
        work = _soft_erode(work)
        delta = F.relu(work - _soft_open(work))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return torch.clamp(skeleton, 0.0, 1.0)


class BinaryClDiceLoss(nn.Module):
    def __init__(self, iterations: int = 20, smooth: float = 1.0):
        super().__init__()
        self.iterations = int(iterations)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        target = target.float()
        if target.ndim == 3:
            target = target.unsqueeze(1)

        skel_pred = soft_skeletonize(probs, iterations=self.iterations)
        skel_true = soft_skeletonize(target, iterations=self.iterations)

        tprec_num = (skel_pred * target).sum(dim=(1, 2, 3))
        tprec_den = skel_pred.sum(dim=(1, 2, 3))
        tsens_num = (skel_true * probs).sum(dim=(1, 2, 3))
        tsens_den = skel_true.sum(dim=(1, 2, 3))

        tprec = (tprec_num + self.smooth) / (tprec_den + self.smooth)
        tsens = (tsens_num + self.smooth) / (tsens_den + self.smooth)
        cl_dice = (2.0 * tprec * tsens) / (tprec + tsens + 1e-8)
        return 1.0 - cl_dice.mean()


def centerline_target_batch(target: torch.Tensor) -> torch.Tensor:
    target = target.float()
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    elif target.ndim != 3:
        raise ValueError(f"Expected target shape (B,H,W) or (B,1,H,W), got {tuple(target.shape)}")

    target_np = (target.detach().cpu().numpy() > 0.5).astype(np.uint8)
    centerlines = []
    for mask in target_np:
        work = mask.copy()
        skeleton = np.zeros_like(work, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(work, element)
            opened = cv2.dilate(eroded, element)
            temp = cv2.subtract(work, opened)
            skeleton = cv2.bitwise_or(skeleton, temp)
            work = eroded
            if cv2.countNonZero(work) == 0:
                break
        centerlines.append(skeleton.astype(np.float32))
    centerline = torch.from_numpy(np.stack(centerlines, axis=0)).to(device=target.device, dtype=torch.float32)
    return centerline.unsqueeze(1)


class BinaryCenterlineDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, centerline_target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        if centerline_target.ndim == 3:
            centerline_target = centerline_target.unsqueeze(1)
        centerline_target = centerline_target.float()

        probs = probs.reshape(probs.shape[0], -1)
        centerline_target = centerline_target.reshape(centerline_target.shape[0], -1)
        intersect = (probs * centerline_target).sum(dim=1)
        denom = probs.sum(dim=1) + centerline_target.sum(dim=1)
        dice = (2 * intersect + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


def calculate_metric_percase(pred, gt):
    pred = np.asarray(pred) > 0
    gt = np.asarray(gt) > 0

    A = pred.sum()
    B = gt.sum()

    if A > 0 and B > 0:
        inter = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        dice = (2 * inter) / (A + B)
        iou = inter / union if union > 0 else 0.0
        precision = inter / A
        recall = inter / B
        return precision, recall, dice, iou
    elif A > 0 and B == 0.0: # For non-crack images
        return 0.0, 0.0, 0.0, 0.0
    elif A == 0.0 and B == 0.0: # For non-crack images
        return 1.0, 1.0, 1.0, 1.0
    elif A == 0.0 and B > 0:
        return 0.0, 0.0, 0.0, 0.0



def test_single_volume(image, label, net, classes, multimask_output, patch_size=[448, 448], input_size=[224, 224],
                       test_save_path=None, case=None, z_spacing=1, boxes=None, points=None,
                       threshold_prob: float = 0.5, use_full_image_box_prompt: bool = False):
    image, label = image.cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy() # image: 1,c,h,w label: h,w
    if len(image.shape) == 4:
        prediction = np.zeros_like(label)
        x, y = image.shape[-2:]
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (1,1,patch_size[0] / x, patch_size[1] / y), order=3)  # ndarray   
        device = next(net.parameters()).device
        inputs = torch.from_numpy(image).float().to(device=device)
        eval_boxes = boxes
        if use_full_image_box_prompt:
            full_h, full_w = image.shape[-2:]
            eval_boxes = torch.tensor(
                [[0.0, 0.0, float(full_w - 1), float(full_h - 1)]],
                dtype=torch.float32,
                device=device,
            )
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0], boxes=eval_boxes, points=points) # inputs 1,c,h,w
            output_masks = outputs['masks']
            if output_masks.shape[1] == 1:
                out = (torch.sigmoid(output_masks) >= float(threshold_prob)).float().squeeze(0).squeeze(0)
            else:
                out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()# h,w  
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0) 

    metric_test=calculate_metric_percase(prediction , label)  # ndarray

    if test_save_path is not None:
        # image: 1,c,h,w  ndarray
        # image = image*255 # Input is already 0-255
        label = label*255
        prediction = prediction*255
        image = Image.fromarray(np.transpose(image.squeeze(0), (1, 2, 0)).astype(np.uint8)) # 1,c,h,w -> h,w,c
        image.save(test_save_path + '/img/' + case + "_img.jpg")
        # pred h,w   ndarray
        pred = Image.fromarray(prediction.astype(np.uint8)) # h,w 
        pred.save(test_save_path + '/pred/' + case + "_img.jpg")
        # label h,w  ndarray
        label = Image.fromarray(label.astype(np.uint8)) # h,w 
        label.save(test_save_path + '/gt/' + case + "_img.jpg")

    return metric_test
