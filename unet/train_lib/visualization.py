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


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(img_tensor):
    """
    Convert a normalized ImageNet tensor (C, H, W) to a denormalized numpy array (H, W, C) in [0, 1].
    """
    # (C, H, W) -> (H, W, C)
    img_np = img_tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    # Un-normalize: x = z * std + mean
    img_np = img_np * IMAGENET_STD + IMAGENET_MEAN
    # Clip to [0, 1]
    return np.clip(img_np, 0.0, 1.0)


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
                images = images.to(device)
                masks = masks.to(device)
                break
        
        if images is None or masks is None:
            print("Visualization skipped: No valid batch found.")
            return

        # Predictions
        # Fix AMP: visualization should also use autocast if training used it, 
        # but for simple inference it usually works fine in fp32. 
        # However, to be safe with model types, we just run forward.
        outputs = model(images)
        predictions = torch.sigmoid(outputs)

        batch_size = images.shape[0]
        n_samples = min(4, batch_size)

        # Panels: Original | GT | Pred Soft | Pred Bin | Overlay (Bin)
        fig, axes = plt.subplots(n_samples, 5, figsize=(20, 5 * n_samples))
        
        # Handle case where n_samples=1 (axes is 1D array)
        if n_samples == 1:
            axes = np.expand_dims(axes, axis=0)

        for i in range(n_samples):
            # Original image (Denormalized)
            img_show = denormalize(images[i])
            img_u8 = (img_show * 255).astype(np.uint8)
            
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
            # Create red overlay
            overlay_mask = np.zeros_like(img_show)
            overlay_mask[..., 0] = 1.0 # Red channel
            
            # Blend: Original + (Red * Mask * Alpha)
            # Efficient overlay: Where mask is 1, blend towards red.
            alpha = 0.5
            mask_expanded = np.expand_dims(pred_bin, axis=-1) # (H,W,1)
            
            # Composite
            composite = img_show * (1 - alpha * mask_expanded) + overlay_mask * (alpha * mask_expanded)
            composite = np.clip(composite, 0.0, 1.0)

            axes[i, 4].imshow((composite * 255).astype(np.uint8))
            axes[i, 4].set_title("Overlay (Bin)")
            axes[i, 4].axis("off")

        plt.tight_layout()
        pred_path = os.path.join(output_dir, f"predictions_epoch_{epoch+1}.png")
        plt.savefig(pred_path)
        # print(f"Prediction visualization saved to {pred_path}") # Reduce log spam
        plt.close()
