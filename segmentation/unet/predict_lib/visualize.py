import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib

    matplotlib.use("Agg")  # Use a headless backend to avoid GUI/Qt issues.
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ModuleNotFoundError:
    plt = None
    _HAS_MPL = False


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


def build_preview_arrays(img: Image.Image, pred: np.ndarray, binary_mask: np.ndarray, max_side=1024):
    return _build_preview(img, pred, binary_mask, max_side=max_side)


def save_prediction_preview(
    output_path: str,
    img_v: np.ndarray,
    pred_v: np.ndarray,
    binary_v: np.ndarray,
    *,
    threshold: float,
    dice: float | None = None,
    gt_mask: np.ndarray | None = None,
    show_gt_panel: bool = False,
):
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
        return

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
