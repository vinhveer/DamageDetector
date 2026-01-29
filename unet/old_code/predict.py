import os
import math
import torch
import numpy as np
from PIL import Image, ImageFilter  # PIL image filtering
from PIL import ImageDraw, ImageFont
import torch.nn.functional as F
try:
    import matplotlib

    matplotlib.use("Agg")  # Use a headless backend to avoid GUI/Qt issues.
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ModuleNotFoundError:
    plt = None
    _HAS_MPL = False
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import argparse
from unet.unet_model import UNet


def _letterbox_with_params(img: Image.Image, size: int, fill, interpolation):
    w, h = img.size
    if w == 0 or h == 0:
        return img, (0, 0, w, h)

    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), resample=interpolation)

    canvas = Image.new(img.mode, (size, size), color=fill)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas, (left, top, new_w, new_h)


def _predict_tiled(model, img: Image.Image, device, tile_size: int, overlap: int, batch_size: int):
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

    img_pad = F.pad(img_tensor, (0, pad_right, 0, pad_bottom), mode="reflect")
    h_pad = h + pad_bottom
    w_pad = w + pad_right

    pred_sum = np.zeros((h_pad, w_pad), dtype=np.float32)
    weight = np.zeros((h_pad, w_pad), dtype=np.float32)

    patches = []
    coords = []

    def _flush():
        if not patches:
            return
        batch = torch.stack(patches, dim=0).to(device)  # (B, C, H, W)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()  # (B, tile, tile)
        for (yy, xx), prob in zip(coords, probs, strict=False):
            pred_sum[yy : yy + tile_size, xx : xx + tile_size] += prob
            weight[yy : yy + tile_size, xx : xx + tile_size] += 1.0
        patches.clear()
        coords.clear()

    for yy in range(0, h_pad - tile_size + 1, step):
        for xx in range(0, w_pad - tile_size + 1, step):
            patch = img_pad[:, yy : yy + tile_size, xx : xx + tile_size]
            patches.append(patch)
            coords.append((yy, xx))
            if len(patches) >= batch_size:
                _flush()
    _flush()

    pred = pred_sum / np.maximum(weight, 1e-8)
    return pred[:h, :w]


def _build_preview(img: Image.Image, pred: np.ndarray, binary_mask: np.ndarray, max_side=1024):
    w, h = img.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img_v = img.resize((new_w, new_h), resample=Image.BILINEAR)
        pred_v = Image.fromarray(pred.astype(np.float32)).resize((new_w, new_h), resample=Image.BILINEAR)
        mask_v = Image.fromarray((binary_mask.astype(np.uint8) * 255)).resize(
            (new_w, new_h), resample=Image.NEAREST
        )
        return np.array(img_v), np.array(pred_v, dtype=np.float32), (np.array(mask_v) > 127)

    return np.array(img), pred, binary_mask


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


def predict_image(
    model,
    image_path,
    device,
    threshold=0.5,
    output_dir="results",
    apply_postprocessing=True,
    output_basename=None,
    gt_mask_path=None,
    gt_expected=False,
    return_details=False,
    roi_box=None,
    mode="tile",
    input_size=256,
    tile_overlap=32,
    tile_batch_size=4,
):
    """Run crack segmentation on a single image using a trained model."""
    # Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image.
    full_img = Image.open(image_path).convert("RGB")
    img = full_img
    img_name = os.path.basename(image_path)
    base_name = output_basename or os.path.splitext(img_name)[0]

    full_size = full_img.size
    if roi_box is not None:
        left, top, right, bottom = [int(v) for v in roi_box]
        left = max(0, min(left, full_size[0]))
        right = max(0, min(right, full_size[0]))
        top = max(0, min(top, full_size[1]))
        bottom = max(0, min(bottom, full_size[1]))
        if right <= left or bottom <= top:
            raise ValueError(f"Invalid roi_box after clamping: {(left, top, right, bottom)}")
        roi_box = (left, top, right, bottom)
        img = full_img.crop(roi_box)

    original_size = img.size
    
    model.eval()

    if mode == "tile":
        pred = _predict_tiled(
            model,
            img,
            device=device,
            tile_size=int(input_size),
            overlap=int(tile_overlap),
            batch_size=int(tile_batch_size),
        )
    elif mode == "letterbox":
        padded, (left, top, new_w, new_h) = _letterbox_with_params(
            img, int(input_size), fill=(0, 0, 0), interpolation=Image.BILINEAR
        )
        img_tensor = transforms.ToTensor()(padded).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            pred_sq = torch.sigmoid(output).squeeze().cpu().numpy()

        pred_crop = pred_sq[top : top + new_h, left : left + new_w]
        pred = np.array(
            Image.fromarray(pred_crop.astype(np.float32)).resize(original_size, resample=Image.BILINEAR),
            dtype=np.float32,
        )
    else:
        # Legacy: stretch to square (fast, but may lose thin cracks).
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (int(input_size), int(input_size)),
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
        pred = np.array(
            Image.fromarray(pred_sq.astype(np.float32)).resize(original_size, resample=Image.BILINEAR),
            dtype=np.float32,
        )
    
    # Apply threshold
    binary_mask = (pred > threshold)
    
    # Post-processing (optional) using PIL filters.
    if apply_postprocessing:
        # Convert NumPy array to PIL image for filtering.
        mask_img = Image.fromarray(binary_mask.astype(np.uint8) * 255).convert("L")
        # Median filter to remove small speckles.
        mask_img = mask_img.filter(ImageFilter.MedianFilter(size=3))
        # Back to NumPy.
        binary_mask = np.array(mask_img) > 127

    if roi_box is not None:
        full_w, full_h = full_size
        left, top, right, bottom = roi_box

        pred_full = np.zeros((full_h, full_w), dtype=np.float32)
        pred_full[top:bottom, left:right] = pred.astype(np.float32)

        binary_full = np.zeros((full_h, full_w), dtype=bool)
        binary_full[top:bottom, left:right] = binary_mask.astype(bool)

        pred = pred_full
        binary_mask = binary_full
        original_size = full_size
        img = full_img
    
    dice = None
    gt_mask = None
    if gt_mask_path:
        try:
            gt_mask = _load_binary_mask(gt_mask_path, target_size=original_size)
            dice = _dice_score(binary_mask, gt_mask)
        except FileNotFoundError:
            gt_mask = None
            dice = None
        except Exception:
            gt_mask = None
            dice = None

    show_gt_panel = bool(gt_expected) or (gt_mask is not None)

    # Save results.
    # Save visualization image.
    output_path = os.path.join(output_dir, f"{base_name}_prediction.png")

    img_v, pred_v, binary_v = _build_preview(img, pred, binary_mask, max_side=1024)
    predmask_v = (binary_v.astype(np.uint8) * 255)
    if gt_mask is not None:
        gt_v = Image.fromarray((gt_mask.astype(np.uint8) * 255)).resize(
            (img_v.shape[1], img_v.shape[0]), resample=Image.NEAREST
        )
        gt_v = np.array(gt_v) > 127
    else:
        gt_v = None

    if _HAS_MPL:
        ncols = 5 if show_gt_panel else 4
        plt.figure(figsize=(5 * ncols, 5))

        plt.subplot(1, ncols, 1)
        plt.imshow(img_v)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, ncols, 2)
        plt.imshow(pred_v, cmap="jet")
        plt.colorbar(label="Crack Probability")
        plt.title(f"Probability Map (Threshold={threshold})")
        plt.axis("off")

        plt.subplot(1, ncols, 3)
        plt.imshow(predmask_v, cmap="gray", vmin=0, vmax=255)
        plt.title("Predicted Mask (Binary)")
        plt.axis("off")

        plt.subplot(1, ncols, 4)
        img_np = img_v.astype(np.float32) / 255.0
        colored_mask = np.zeros_like(img_np)
        colored_mask[..., 0] = binary_v.astype(np.float32)
        overlay = img_np * 0.7 + colored_mask * 0.3
        overlay = np.clip(overlay, 0, 1)

        plt.imshow(overlay)
        if dice is None:
            plt.title("Overlay Display")
        else:
            plt.title(f"Overlay Display\nDice={dice:.4f}")
        plt.axis("off")

        if show_gt_panel:
            plt.subplot(1, ncols, 5)
            if gt_v is None:
                plt.imshow(np.zeros((img_v.shape[0], img_v.shape[1]), dtype=np.uint8), cmap="gray")
                plt.text(
                    0.5,
                    0.5,
                    "GT mask not found",
                    color="white",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                )
                plt.title("Ground Truth Mask (Missing)")
            else:
                plt.imshow(gt_v, cmap="gray")
                plt.title("Ground Truth Mask")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
    else:
        # Fallback when matplotlib is not installed: save a multi-panel PNG via PIL.
        img_resized = Image.fromarray(img_v).convert("RGB")

        def _jet_rgb(x):
            x = np.clip(x, 0.0, 1.0)
            r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
            g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
            b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
            return np.stack([r, g, b], axis=-1)

        heatmap_rgb = (_jet_rgb(pred_v) * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap_rgb).convert("RGB")

        img_np = np.array(img_resized).astype(np.float32) / 255.0
        colored_mask = np.zeros_like(img_np)
        colored_mask[..., 0] = binary_v.astype(np.float32)
        overlay = img_np * 0.7 + colored_mask * 0.3
        overlay_img = Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8))

        w, h = img_resized.size
        ncols = 5 if show_gt_panel else 4
        panel = Image.new("RGB", (w * ncols, h))
        panel.paste(img_resized, (0, 0))
        panel.paste(heatmap_img, (w, 0))
        predmask_img = Image.fromarray(predmask_v).convert("RGB")
        panel.paste(predmask_img, (w * 2, 0))
        panel.paste(overlay_img, (w * 3, 0))
        if show_gt_panel:
            if gt_v is None:
                gt_img = Image.new("RGB", (w, h), color=(0, 0, 0))
                draw_gt = ImageDraw.Draw(gt_img)
                try:
                    font_gt = ImageFont.load_default()
                except Exception:
                    font_gt = None
                draw_gt.text((10, 10), "GT mask not found", fill=(255, 255, 255), font=font_gt)
            else:
                gt_img = Image.fromarray((gt_v.astype(np.uint8) * 255)).convert("RGB")
            panel.paste(gt_img, (w * 4, 0))

        if dice is not None:
            draw = ImageDraw.Draw(panel)
            text = f"Dice={dice:.4f}"
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            x0 = w * 3 + 5
            draw.rectangle([x0, 5, x0 + 120, 25], fill=(0, 0, 0))
            draw.text((x0 + 5, 8), text, fill=(255, 255, 255), font=font)
        panel.save(output_path)
    
    # Save the binary mask (post-processed if enabled).
    binary_mask_out = binary_mask.astype(np.uint8) * 255
    mask_output = Image.fromarray(binary_mask_out).resize(original_size, resample=Image.NEAREST)
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    mask_output.save(mask_path)
    
    print(f"Prediction saved to: {output_path}")
    print(f"Binary mask saved to: {mask_path}")
    if gt_mask_path and dice is not None:
        print(f"Ground truth: {gt_mask_path}")
        print(f"Dice: {dice:.4f}")
    elif gt_expected:
        if gt_mask_path:
            print(f"Warning: GT mask not found/invalid: {gt_mask_path}")
        else:
            print("Warning: GT mask not provided (gt_expected=True)")
    if return_details:
        return {
            "output_path": output_path,
            "mask_path": mask_path,
            "dice": dice,
            "gt_mask_path": gt_mask_path,
            "roi_box": roi_box,
        }
    return output_path

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description='Run crack prediction using a trained model.')
    parser.add_argument('--image', type=str, required=True, help='Input image path.')
    parser.add_argument('--model', type=str, default='output_results/best_model.pth', help='Model weights path.')
    parser.add_argument('--output', type=str, default='results', help='Output directory.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binarization threshold in [0, 1].')
    parser.add_argument('--no-postprocessing', action='store_true', help='Disable post-processing.')
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tile", "letterbox", "resize"],
        default="tile",
        help="tile: sliding-window over original image; letterbox: keep aspect ratio; resize: stretch to square.",
    )
    parser.add_argument("--input-size", type=int, default=256, help="Model input size (square).")
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=0,
        help="Overlap (pixels) between tiles in tile mode. 0 means input_size//2 (recommended).",
    )
    parser.add_argument("--tile-batch-size", type=int, default=4, help="Batch size for tiles in tile mode.")
    args = parser.parse_args()
    
    # Check files exist.
    if not os.path.exists(args.image):
        print(f"Error: image file does not exist: {args.image}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: model file does not exist: {args.model}")
        return
    
    # Select device.
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load model.
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    print(f"Loaded model: {args.model}")
    
    # Predict
    predict_image(
        model, 
        args.image, 
        device, 
        threshold=args.threshold,
        output_dir=args.output,
        apply_postprocessing=not args.no_postprocessing,
        mode=args.mode,
        input_size=args.input_size,
        tile_overlap=(args.input_size // 2) if args.tile_overlap == 0 else args.tile_overlap,
        tile_batch_size=args.tile_batch_size,
    )

if __name__ == '__main__':
    main() 
