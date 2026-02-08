import torch
import os
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

# --- Imports logic ---
from train_lib.cli import build_arg_parser, load_config
from train_lib.runner import collate_skip_none
from dataset_lib import CrackDataset
from dataset_lib.utils import build_mask_index
from train_lib.metrics import iou_score_from_prob


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Best Model Checkpoint
    model_path = os.path.join(args.output_dir, "best_model.pth")
    if not os.path.exists(model_path):
        print("Best model not found.")
        return

    # Initialize Model Structure
    model = smp.UnetPlusPlus(
        encoder_name=getattr(args, "encoder_name", "efficientnet-b0"),
        encoder_weights=None, # Weights from checkpoint
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)
    
    # Load State Dict
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Val Dataset
    val_mask_index = build_mask_index(args.val_masks)
    val_names = CrackDataset.list_valid_images(
        args.val_images,
        args.val_masks,
        mask_prefix=args.mask_prefix,
        mask_index=val_mask_index,
    )
    val_dataset = CrackDataset(
        image_dir=args.val_images,
        mask_dir=args.val_masks,
        image_filenames=val_names,
        mask_prefix=args.mask_prefix,
        mask_index=val_mask_index,
        augment=False,
        output_size=args.input_size,
        preprocess_mode="letterbox" 
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_skip_none,
        pin_memory=True
    )

    print(f"Checking thresholds on {len(val_dataset)} images...")

    thresholds = torch.arange(0.1, 0.95, 0.05).to(device)
    iou_sums = torch.zeros(len(thresholds)).to(device)
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            if batch is None: continue
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device) # (B, 1, H, W)

            # Forward
            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(images)
                probs = torch.sigmoid(outputs) # (B, 1, H, W)

            # Batch IoU for all thresholds efficiently
            # Expand probs to (T, B, 1, H, W) vs masks
            # Too heavy memory? Iterate thresholds
            batch_ious = []
            for i, thr in enumerate(thresholds):
                batch_ious.append(iou_score_from_prob(probs, masks, thr=float(thr)))
            
            iou_sums += torch.tensor(batch_ious).to(device)
            n_batches += 1

    avg_ious = iou_sums / n_batches
    
    print("\n--- Evaluation Results ---")
    best_thr_idx = torch.argmax(avg_ious)
    best_iou = avg_ious[best_thr_idx].item()
    best_thr = thresholds[best_thr_idx].item()

    for t, iou in zip(thresholds, avg_ious):
        mark = "🏆" if t == best_thr else ""
        print(f"Thr: {t:.2f} | IoU: {iou:.4f} {mark}")

    print("-" * 30)
    print(f"Best Threshold: {best_thr:.2f}")
    print(f"Best IoU:       {best_iou:.4f}")

if __name__ == "__main__":
    main()
