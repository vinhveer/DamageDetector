import math
import logging

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from torch_runtime import get_torch, get_torch_nn_functional

from .preprocess import _letterbox_with_params
from .types import StopRequested


LOGGER = logging.getLogger(__name__)


def _mask_logits(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _make_gaussian_weight(tile_size: int, sigma_ratio: float = 0.25) -> np.ndarray:
    sigma = float(tile_size) * float(sigma_ratio)
    coords = np.arange(tile_size, dtype=np.float32) - (tile_size - 1) / 2.0
    gauss_1d = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    weight = gauss_1d[:, None] * gauss_1d[None, :]
    return (weight / np.max(weight)).astype(np.float32)


def _parse_scales(scales):
    if scales is None:
        return (1.0,)
    if isinstance(scales, str):
        values = [float(x.strip()) for x in scales.split(",") if x.strip()]
    else:
        values = [float(x) for x in scales]
    values = tuple(x for x in values if x > 0)
    return values or (1.0,)


def _resize_prob(prob: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(prob.astype(np.float32), size, interpolation=cv2.INTER_LINEAR).astype(np.float32)


def _predict_tiled(
    model,
    img: Image.Image,
    device,
    tile_size: int,
    overlap: int,
    batch_size: int,
    *,
    gaussian_weight=False,
    stop_checker=None,
):
    torch = get_torch()
    F = get_torch_nn_functional()
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be in [0, tile_size)")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img)  # (C, H, W)
    _, h, w = img_tensor.shape
    step = tile_size - overlap

    n_y = 1 if h <= tile_size else (math.ceil((h - tile_size) / step) + 1)
    n_x = 1 if w <= tile_size else (math.ceil((w - tile_size) / step) + 1)
    padded_h = (n_y - 1) * step + tile_size
    padded_w = (n_x - 1) * step + tile_size
    pad_bottom = max(0, padded_h - h)
    pad_right = max(0, padded_w - w)

    img_pad = F.pad(img_tensor, (0, pad_right, 0, pad_bottom), mode="replicate")
    h_pad = h + pad_bottom
    w_pad = w + pad_right

    pred_sum = np.zeros((h_pad, w_pad), dtype=np.float32)
    weight = np.zeros((h_pad, w_pad), dtype=np.float32)
    tile_weight = _make_gaussian_weight(tile_size) if gaussian_weight else None

    patches = []
    coords = []

    def _flush():
        if not patches:
            return
        if stop_checker is not None and stop_checker():
            raise StopRequested("Stopped")
        batch = torch.stack(patches, dim=0).to(device)  # (B, C, H, W)
        with torch.no_grad():
            logits = _mask_logits(model(batch))
            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()  # (B, tile, tile)
        for (yy, xx), prob in zip(coords, probs, strict=False):
            if tile_weight is None:
                pred_sum[yy : yy + tile_size, xx : xx + tile_size] += prob
                weight[yy : yy + tile_size, xx : xx + tile_size] += 1.0
            else:
                pred_sum[yy : yy + tile_size, xx : xx + tile_size] += prob * tile_weight
                weight[yy : yy + tile_size, xx : xx + tile_size] += tile_weight
        patches.clear()
        coords.clear()

    y_steps = range(0, h_pad - tile_size + 1, step)
    x_steps = range(0, w_pad - tile_size + 1, step)
    total_tiles = len(y_steps) * len(x_steps)
    processed_tiles = 0

    LOGGER.debug("UNet tiled inference: processing %s tiles on %s", total_tiles, device)

    for yy in y_steps:
        for xx in x_steps:
            if stop_checker is not None and stop_checker():
                LOGGER.debug("UNet tiled inference stopped by stop checker")
                raise StopRequested("Stopped")
            patch = img_pad[:, yy : yy + tile_size, xx : xx + tile_size]
            patches.append(patch)
            coords.append((yy, xx))
            if len(patches) >= batch_size:
                processed_tiles += len(patches)
                _flush()
                LOGGER.debug("UNet tiled inference: processed %s/%s tiles", processed_tiles, total_tiles)

    remaining = len(patches)
    _flush()
    if remaining:
        processed_tiles += remaining
        LOGGER.debug("UNet tiled inference: processed %s/%s tiles", processed_tiles, total_tiles)

    pred = pred_sum / np.maximum(weight, 1e-8)
    return pred[:h, :w]


def predict_probabilities(
    model,
    img: Image.Image,
    device,
    *,
    mode="tile",
    input_size=256,
    tile_overlap=128,
    tile_batch_size=4,
    tta=False,
    multiscale=None,
    gaussian_weight=False,
    stop_checker=None,
):
    """Return a float32 probability map (H, W) in [0, 1]."""
    torch = get_torch()
    model.eval()
    mode = str(mode).lower()
    input_size = int(input_size)
    scales = _parse_scales(multiscale)

    if len(scales) > 1 or scales[0] != 1.0:
        original_size = img.size
        probs = []
        for scale in scales:
            scaled = img if scale == 1.0 else img.resize(
                (max(1, int(round(original_size[0] * scale))), max(1, int(round(original_size[1] * scale)))),
                Image.BILINEAR,
            )
            prob = predict_probabilities(
                model,
                scaled,
                device,
                mode=mode,
                input_size=input_size,
                tile_overlap=tile_overlap,
                tile_batch_size=tile_batch_size,
                tta=tta,
                multiscale=(1.0,),
                gaussian_weight=gaussian_weight,
                stop_checker=stop_checker,
            )
            if scaled.size != original_size:
                prob = _resize_prob(prob, original_size)
            probs.append(prob)
        return np.mean(np.stack(probs, axis=0), axis=0).astype(np.float32)

    if tta:
        transforms_list = [
            (lambda x: x, lambda p: p),
            (lambda x: x.transpose(Image.FLIP_LEFT_RIGHT), lambda p: np.fliplr(p)),
            (lambda x: x.transpose(Image.FLIP_TOP_BOTTOM), lambda p: np.flipud(p)),
            (lambda x: x.rotate(180), lambda p: np.rot90(p, 2)),
        ]
        probs = []
        for forward, inverse in transforms_list:
            prob = predict_probabilities(
                model,
                forward(img),
                device,
                mode=mode,
                input_size=input_size,
                tile_overlap=tile_overlap,
                tile_batch_size=tile_batch_size,
                tta=False,
                multiscale=(1.0,),
                gaussian_weight=gaussian_weight,
                stop_checker=stop_checker,
            )
            probs.append(np.ascontiguousarray(inverse(prob)))
        return np.mean(np.stack(probs, axis=0), axis=0).astype(np.float32)

    if mode == "tile":
        return _predict_tiled(
            model,
            img,
            device=device,
            tile_size=input_size,
            overlap=int(tile_overlap),
            batch_size=int(tile_batch_size),
            gaussian_weight=bool(gaussian_weight),
            stop_checker=stop_checker,
        )

    if mode == "letterbox":
        if stop_checker is not None and stop_checker():
            raise StopRequested("Stopped")
        original_size = img.size
        padded, (left, top, new_w, new_h) = _letterbox_with_params(
            img, input_size, fill=(0, 0, 0), interpolation=Image.BILINEAR
        )
        img_tensor = transforms.ToTensor()(padded).unsqueeze(0).to(device)
        with torch.no_grad():
            output = _mask_logits(model(img_tensor))
            pred_sq = torch.sigmoid(output).squeeze().cpu().numpy()

        pred_crop = pred_sq[top : top + new_h, left : left + new_w]
        return _resize_prob(pred_crop, original_size)

    if stop_checker is not None and stop_checker():
        raise StopRequested("Stopped")
    transform = transforms.Compose(
        [
            transforms.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
        ]
    )
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = _mask_logits(model(img_tensor))
        pred_sq = torch.sigmoid(output).squeeze().cpu().numpy()
    return _resize_prob(pred_sq, img.size)
