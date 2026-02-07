import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .preprocess import _letterbox_with_params
from .types import StopRequested


def _predict_tiled(
    model,
    img: Image.Image,
    device,
    tile_size: int,
    overlap: int,
    batch_size: int,
    *,
    stop_checker=None,
):
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be in [0, tile_size)")

    img_tensor = transforms.ToTensor()(img)  # (C, H, W)
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

    patches = []
    coords = []

    def _flush():
        if not patches:
            return
        if stop_checker is not None and stop_checker():
            raise StopRequested("Stopped")
        batch = torch.stack(patches, dim=0).to(device)  # (B, C, H, W)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()  # (B, tile, tile)
        for (yy, xx), prob in zip(coords, probs, strict=False):
            pred_sum[yy : yy + tile_size, xx : xx + tile_size] += prob
            weight[yy : yy + tile_size, xx : xx + tile_size] += 1.0
        patches.clear()
        coords.clear()

    y_steps = range(0, h_pad - tile_size + 1, step)
    x_steps = range(0, w_pad - tile_size + 1, step)
    total_tiles = len(y_steps) * len(x_steps)
    processed_tiles = 0

    print(f"DEBUG: Processing {total_tiles} tiles on {device}...")

    for yy in y_steps:
        for xx in x_steps:
            if stop_checker is not None and stop_checker():
                print("Inference: Stop checker triggered!")
                raise StopRequested("Stopped")
            patch = img_pad[:, yy : yy + tile_size, xx : xx + tile_size]
            patches.append(patch)
            coords.append((yy, xx))
            if len(patches) >= batch_size:
                _flush()
                processed_tiles += len(patches)
                # Print progress every batch or so
                print(f"UNet: processed {processed_tiles}/{total_tiles} tiles...")
                
    _flush()
    if processed_tiles < total_tiles:
         print(f"UNet: finished remaining {total_tiles - processed_tiles} tiles.")

    pred = pred_sum / np.maximum(weight, 1e-8)
    return pred[:h, :w]


def predict_probabilities(
    model,
    img: Image.Image,
    device,
    *,
    mode="tile",
    input_size=256,
    tile_overlap=32,
    tile_batch_size=4,
    stop_checker=None,
):
    """Return a float32 probability map (H, W) in [0, 1]."""
    model.eval()
    mode = str(mode).lower()
    input_size = int(input_size)

    if mode == "tile":
        return _predict_tiled(
            model,
            img,
            device=device,
            tile_size=input_size,
            overlap=int(tile_overlap),
            batch_size=int(tile_batch_size),
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
            output = model(img_tensor)
            pred_sq = torch.sigmoid(output).squeeze().cpu().numpy()

        pred_crop = pred_sq[top : top + new_h, left : left + new_w]
        return np.array(
            Image.fromarray(pred_crop.astype(np.float32)).resize(original_size, resample=Image.BILINEAR),
            dtype=np.float32,
        )

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
        output = model(img_tensor)
        pred_sq = torch.sigmoid(output).squeeze().cpu().numpy()
    return np.array(
        Image.fromarray(pred_sq.astype(np.float32)).resize(img.size, resample=Image.BILINEAR),
        dtype=np.float32,
    )
