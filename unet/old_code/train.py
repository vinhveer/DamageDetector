import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import numpy as np
from PIL import Image
try:
    import matplotlib

    matplotlib.use("Agg")  # Headless backend (no GUI required).
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ModuleNotFoundError:
    plt = None
    _HAS_MPL = False
from unet.unet_model import UNet
from dataset import CrackDataset, LetterboxResize, RandomPatchDataset, TiledDataset
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use only the second GPU

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)

def dice_loss(pred, target):
    smooth = 1.0
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, output_dir):
    # Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    best_val_dice = -1.0
    train_losses = []
    val_losses = []
    # Early stopping parameters.
    patience = 15
    counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            if batch is None:
                continue
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backprop + optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        if train_steps == 0:
            print("Warning: no valid training batches in this epoch (all samples failed to load).")
            avg_train_loss = float("nan")
        else:
            avg_train_loss = train_loss / train_steps
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        val_dice_sum = 0.0
        val_iou_sum = 0.0
        val_thr_report = {}
        if hasattr(train_model, "_metric_thresholds"):
            for t in train_model._metric_thresholds:
                val_thr_report[t] = {"dice": 0.0, "iou": 0.0}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                if batch is None:
                    continue
                images, masks = batch
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_steps += 1
                thr = getattr(train_model, "_metric_threshold", 0.5)
                val_dice_sum += dice_score(outputs, masks, thr=thr)
                val_iou_sum += iou_score(outputs, masks, thr=thr)

                if val_thr_report:
                    for t in val_thr_report.keys():
                        val_thr_report[t]["dice"] += dice_score(outputs, masks, thr=float(t))
                        val_thr_report[t]["iou"] += iou_score(outputs, masks, thr=float(t))
        
        if val_steps == 0:
            print("Warning: no valid validation batches (all samples failed to load).")
            avg_val_loss = float("inf")
            avg_val_dice = 0.0
            avg_val_iou = 0.0
        else:
            avg_val_loss = val_loss / val_steps
            avg_val_dice = val_dice_sum / val_steps
            avg_val_iou = val_iou_sum / val_steps
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        thr = getattr(train_model, "_metric_threshold", 0.5)
        print(f'Val Loss: {avg_val_loss:.4f} | Val Dice@{thr}: {avg_val_dice:.4f} | Val IoU@{thr}: {avg_val_iou:.4f}')
        if val_steps > 0 and val_thr_report:
            items = []
            for t in sorted(val_thr_report.keys()):
                d = val_thr_report[t]["dice"] / val_steps
                j = val_thr_report[t]["iou"] / val_steps
                items.append(f"{t}:D{d:.3f}/I{j:.3f}")
            print("Val metrics sweep (thr:Dice/IoU): " + " | ".join(items))
        
        # Step the learning-rate scheduler.
        sched_metric = getattr(train_model, "_scheduler_metric", "loss")
        scheduler.step(avg_val_dice if sched_metric == "dice" else avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr:.7f}')
        
        # Save the best model (by validation Dice, not loss).
        # For thin cracks, loss can decrease by over-predicting (fatter masks) while Dice/IoU worsens.
        if val_steps > 0 and avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_val_loss = avg_val_loss  # tracked for reference/logging only
            counter = 0  # Reset early-stopping counter
            model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Best model (Val Dice) saved to {model_path}!')
        else:
            counter += 1
            print(f'Val Dice did not improve. Early stopping counter: {counter}/{patience}')
            if counter >= patience:
                print(f'Early stopping! No Val Dice improvement for {patience} consecutive epochs.')
                break
        
        # Visualize a few predictions.
        if _HAS_MPL and (epoch + 1) % 5 == 0:
            visualize_predictions(model, val_loader, device, epoch, output_dir, thr=thr)
    
    # Plot the loss curves.
    if _HAS_MPL:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        loss_curve_path = os.path.join(output_dir, 'loss_curve.png')
        plt.savefig(loss_curve_path)
        print(f'Loss curve saved to {loss_curve_path}')
        plt.close()
    else:
        print("matplotlib is not installed; skipping loss curves and prediction visualizations.")

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
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # Ground truth
            mask = masks[i].cpu().numpy().squeeze()
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Prediction (soft)
            pred_soft = predictions[i].cpu().numpy().squeeze()
            axes[i, 2].imshow(pred_soft, cmap='gray', vmin=0.0, vmax=1.0)
            axes[i, 2].set_title('Pred Soft')
            axes[i, 2].axis('off')

            # Prediction (binary @ thr)
            pred_bin = (pred_soft > thr).astype(np.float32)
            axes[i, 3].imshow(pred_bin, cmap='gray', vmin=0.0, vmax=1.0)
            axes[i, 3].set_title(f'Pred Bin@{thr}')
            axes[i, 3].axis('off')

            # Overlay binary mask on original image (red crack)
            alpha = 0.35
            red = np.zeros_like(img_f)
            red[..., 0] = pred_bin
            overlay = np.clip(img_f * (1.0 - alpha) + red * alpha, 0.0, 1.0)
            axes[i, 4].imshow((overlay * 255).astype(np.uint8))
            axes[i, 4].set_title('Overlay (Bin)')
            axes[i, 4].axis('off')
        
        plt.tight_layout()
        pred_path = os.path.join(output_dir, f'predictions_epoch_{epoch+1}.png')
        plt.savefig(pred_path)
        print(f'Prediction visualization saved to {pred_path}')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train U-Net for crack segmentation.")
    parser.add_argument(
        "--train-images",
        type=str,
        required=True,
        help="Train images folder.",
    )
    parser.add_argument(
        "--train-masks",
        type=str,
        required=True,
        help="Train masks folder.",
    )
    parser.add_argument(
        "--val-images",
        type=str,
        required=True,
        help="Validation images folder.",
    )
    parser.add_argument(
        "--val-masks",
        type=str,
        required=True,
        help="Validation masks folder.",
    )
    parser.add_argument(
        "--mask-prefix",
        type=str,
        default="auto",
        help="Mask filename suffix appended to image base name (before extension). "
        "Example: image 'abc.jpg' -> mask 'abc{mask_prefix}.*'. Use '' for identical base names. "
        "Use 'auto' to try both '' and '_mask'.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output_results", help="Output directory."
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        choices=["patch", "letterbox", "stretch"],
        default="patch",
        help="patch: crop 256x256 at original scale; letterbox: keep aspect ratio with padding; stretch: legacy Resize((H,W)).",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=512,
        help="Model input size (square). Use multiples of 16 (e.g., 256, 512).",
    )
    parser.add_argument(
        "--patches-per-image",
        type=int,
        default=16,
        help="(patch mode) Number of random patches sampled per image (virtual length multiplier).",
    )
    parser.add_argument(
        "--max-patch-tries",
        type=int,
        default=10,
        help="(patch mode) Try K times to find a patch containing cracks before falling back to random.",
    )
    parser.add_argument(
        "--val-stride",
        type=int,
        default=0,
        help="(patch mode) Stride for val tiling. 0 means input_size//2 (overlap).",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers.")
    parser.add_argument("--epochs", type=int, default=80, help="Number of epochs.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument(
        "--no-augment", action="store_true", help="Disable data augmentation."
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=5.0,
        help="Positive class weight for BCEWithLogits (tune ~3-10 for class imbalance).",
    )
    parser.add_argument(
        "--bce-weight",
        type=float,
        default=0.4,
        help="Weight for BCEWithLogits term in the total loss.",
    )
    parser.add_argument(
        "--dice-weight",
        type=float,
        default=0.6,
        help="Weight for Dice loss term in the total loss.",
    )
    parser.add_argument(
        "--metric-threshold",
        type=float,
        default=0.5,
        help="Threshold used to compute Val Dice/IoU metrics.",
    )
    parser.add_argument(
        "--metric-thresholds",
        type=str,
        default="",
        help="Optional comma-separated thresholds to report (e.g. '0.3,0.5,0.7').",
    )
    parser.add_argument(
        "--scheduler-metric",
        type=str,
        choices=["loss", "dice"],
        default="loss",
        help="Metric to drive ReduceLROnPlateau (loss is stable default; dice may better reflect thin-crack quality).",
    )
    args = parser.parse_args()

    # Select device.
    device = torch.device("cpu")
    print(f'Using device: {device}')
    
    # Output directory.
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f'Output directory: {output_dir}')
    
    # Dataset paths.
    train_images_path = args.train_images
    train_masks_path = args.train_masks
    val_images_path = args.val_images
    val_masks_path = args.val_masks

    train_names = CrackDataset.list_valid_images(
        train_images_path, train_masks_path, mask_prefix=args.mask_prefix
    )
    val_names = CrackDataset.list_valid_images(
        val_images_path, val_masks_path, mask_prefix=args.mask_prefix
    )
    if not train_names:
        raise RuntimeError(
            f"No valid image-mask pairs found under: {train_images_path} and {train_masks_path}"
        )
    if not val_names:
        raise RuntimeError(
            f"No valid image-mask pairs found under: {val_images_path} and {val_masks_path}"
        )

    # Preprocessing / datasets
    if args.preprocess == "patch":
        image_transform = transforms.Compose([transforms.ToTensor()])
        mask_transform = transforms.Compose([transforms.ToTensor()])

        train_dataset = RandomPatchDataset(
            image_dir=train_images_path,
            mask_dir=train_masks_path,
            image_filenames=train_names,
            mask_prefix=args.mask_prefix,
            patch_size=args.input_size,
            patches_per_image=args.patches_per_image,
            max_patch_tries=args.max_patch_tries,
            augment=not args.no_augment,
            image_transform=image_transform,
            mask_transform=mask_transform,
            verbose=True,
        )

        stride = args.val_stride if args.val_stride and args.val_stride > 0 else args.input_size // 2
        val_dataset = TiledDataset(
            image_dir=val_images_path,
            mask_dir=val_masks_path,
            image_filenames=val_names,
            mask_prefix=args.mask_prefix,
            patch_size=args.input_size,
            stride=stride,
            image_transform=image_transform,
            mask_transform=mask_transform,
            verbose=True,
        )

    elif args.preprocess == "letterbox":
        # Keep aspect ratio: resize to fit, then pad to square.
        image_transform = transforms.Compose(
            [
                LetterboxResize(args.input_size, fill=(0, 0, 0), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        mask_transform = transforms.Compose(
            [
                LetterboxResize(args.input_size, fill=0, interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ]
        )
        train_full = CrackDataset(
            image_dir=train_images_path,
            mask_dir=train_masks_path,
            image_filenames=train_names,
            mask_prefix=args.mask_prefix,
            image_transform=image_transform,
            mask_transform=mask_transform,
            augment=not args.no_augment,
            patch_size=None,
            output_size=args.input_size,
            verbose=True,
        )
        val_full = CrackDataset(
            image_dir=val_images_path,
            mask_dir=val_masks_path,
            image_filenames=val_names,
            mask_prefix=args.mask_prefix,
            image_transform=image_transform,
            mask_transform=mask_transform,
            augment=False,
            patch_size=None,
            output_size=args.input_size,
            verbose=True,
        )
        train_dataset = train_full
        val_dataset = val_full
    else:
        # Legacy behaviour: stretch to square (kept for comparison).
        image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (args.input_size, args.input_size),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.ToTensor(),
            ]
        )
        mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (args.input_size, args.input_size),
                    interpolation=InterpolationMode.NEAREST,
                    antialias=False,
                ),
                transforms.ToTensor(),
            ]
        )
        train_full = CrackDataset(
            image_dir=train_images_path,
            mask_dir=train_masks_path,
            image_filenames=train_names,
            mask_prefix=args.mask_prefix,
            image_transform=image_transform,
            mask_transform=mask_transform,
            augment=not args.no_augment,
            patch_size=None,
            output_size=args.input_size,
            verbose=True,
        )
        val_full = CrackDataset(
            image_dir=val_images_path,
            mask_dir=val_masks_path,
            image_filenames=val_names,
            mask_prefix=args.mask_prefix,
            image_transform=image_transform,
            mask_transform=mask_transform,
            augment=False,
            patch_size=None,
            output_size=args.input_size,
            verbose=True,
        )
        train_dataset = train_full
        val_dataset = val_full
    
    print(f'Train set size: {len(train_dataset)}')
    print(f'Val set size: {len(val_dataset)}')
    
    # Data loaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_skip_none,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_skip_none,
    )
    
    # Model.
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    # Loss + optimizer.
    pos_weight = torch.tensor([float(args.pos_weight)], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def criterion(pred, target):
        return float(args.bce_weight) * bce(pred, target) + float(args.dice_weight) * dice_loss(pred, target)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    # LR scheduler.
    sched_mode = "max" if args.scheduler_metric == "dice" else "min"
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=sched_mode, factor=0.5, patience=10, verbose=True
        )
    except TypeError:
        # Newer PyTorch versions removed the 'verbose' argument.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=sched_mode, factor=0.5, patience=10
        )

    if args.preprocess == "patch":
        print(
            f"Patch training: patch_size={args.input_size}, train_patches_per_image={args.patches_per_image}, "
            f"val_stride={args.val_stride if args.val_stride > 0 else (args.input_size // 2)}"
        )
    print(
        f"Loss: bce_weight={args.bce_weight}, dice_weight={args.dice_weight}, pos_weight={args.pos_weight}"
    )
    print(f"Val metric threshold: {args.metric_threshold} | Scheduler metric: {args.scheduler_metric}")

    train_model._metric_threshold = float(args.metric_threshold)
    train_model._scheduler_metric = args.scheduler_metric
    if args.metric_thresholds.strip():
        train_model._metric_thresholds = [
            float(x.strip()) for x in args.metric_thresholds.split(",") if x.strip()
        ]
    else:
        train_model._metric_thresholds = []
    
    # Train.
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,  # Pass the scheduler
        num_epochs=args.epochs,
        device=device,
        output_dir=output_dir
    )

if __name__ == '__main__':
    main() 
