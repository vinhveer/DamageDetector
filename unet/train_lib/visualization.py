import os

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")  # Headless backend (no GUI required).
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ModuleNotFoundError:
    plt = None
    _HAS_MPL = False


def visualize_predictions(model, val_loader, device, epoch, output_dir, thr=0.5):
    if not _HAS_MPL:
        return
    model.eval()  # Evaluation mode
    with torch.no_grad():
        # Get one validation batch (skip None batches).
        images, masks = None, None
        for batch in val_loader:
            if batch is None:
                continue
            images, masks = batch
            if images is not None and masks is not None:
                break
        if images is None or masks is None:
            return
        images = images.to(device)
        masks = masks.to(device)

        # Predictions
        outputs = model(images)
        predictions = torch.sigmoid(outputs)

        # Visualize the first 4 samples.
        # Panels: Original | GT | Pred Soft | Pred Bin | Overlay (Bin)
        fig, axes = plt.subplots(4, 5, figsize=(24, 20))

        for i in range(4):
            # Original image
            img_f = images[i].detach().cpu().float().numpy().transpose(1, 2, 0)
            img_f = np.clip(img_f, 0.0, 1.0)
            img_u8 = (img_f * 255).astype(np.uint8)
            axes[i, 0].imshow(img_u8)
            axes[i, 0].set_title("Original Image")
            axes[i, 0].axis("off")

            # Ground truth
            mask = masks[i].cpu().numpy().squeeze()
            axes[i, 1].imshow(mask, cmap="gray")
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis("off")

            # Prediction (soft)
            pred_soft = predictions[i].cpu().numpy().squeeze()
            axes[i, 2].imshow(pred_soft, cmap="gray", vmin=0.0, vmax=1.0)
            axes[i, 2].set_title("Pred Soft")
            axes[i, 2].axis("off")

            # Prediction (binary @ thr)
            pred_bin = (pred_soft > thr).astype(np.float32)
            axes[i, 3].imshow(pred_bin, cmap="gray", vmin=0.0, vmax=1.0)
            axes[i, 3].set_title(f"Pred Bin@{thr}")
            axes[i, 3].axis("off")

            # Overlay binary mask on original image (red crack)
            alpha = 0.35
            red = np.zeros_like(img_f)
            red[..., 0] = pred_bin
            overlay = np.clip(img_f * (1.0 - alpha) + red * alpha, 0.0, 1.0)
            axes[i, 4].imshow((overlay * 255).astype(np.uint8))
            axes[i, 4].set_title("Overlay (Bin)")
            axes[i, 4].axis("off")

        plt.tight_layout()
        pred_path = os.path.join(output_dir, f"predictions_epoch_{epoch+1}.png")
        plt.savefig(pred_path)
        print(f"Prediction visualization saved to {pred_path}")
        plt.close()
