import csv
import logging
import os
import sys
import random
from datetime import datetime

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

from dataset_lib import (
    CrackDataset,
    LetterboxResize,
    RandomPatchDataset,
    TiledDataset,
    build_mask_index,
    find_mask_path,
)
from .collate import collate_skip_none
from .losses import dice_loss, focal_loss_with_logits
from .training import train_model


def _estimate_pos_weight(
    train_names,
    mask_dir,
    mask_prefix,
    mask_index,
    sample_size=200,
    min_weight=1.0,
    max_weight=20.0,
):
    if not train_names:
        return None
    names = train_names
    if sample_size and len(train_names) > sample_size:
        names = random.sample(train_names, int(sample_size))

    pos = 0
    total = 0
    used = 0
    for img_name in names:
        base_name = os.path.splitext(img_name)[0]
        mask_path = find_mask_path(
            mask_dir,
            base_name,
            mask_prefix=mask_prefix,
            mask_index=mask_index,
        )
        if mask_path is None:
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        pos += int((mask > 0).sum())
        total += int(mask.size)
        used += 1

    if total == 0:
        return None

    pos_ratio = pos / float(total)
    if pos_ratio <= 0:
        weight = float(max_weight)
    else:
        weight = (1.0 - pos_ratio) / pos_ratio
    weight = max(float(min_weight), min(float(max_weight), float(weight)))
    return float(weight), float(pos_ratio), int(used)


def run_training(args):
    # Select device.
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            device = torch.device("cuda")
        else:
            device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    
    print(f'Using device: {device}')

    # Output directory.
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if output_dir:
        base = os.path.basename(os.path.normpath(output_dir))
        if base == "output_results":
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(output_dir, stamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f'Output directory: {output_dir}')

    # Dataset paths.
    train_images_path = args.train_images
    train_masks_path = args.train_masks
    val_images_path = args.val_images
    val_masks_path = args.val_masks

    train_mask_index = build_mask_index(train_masks_path)
    val_mask_index = build_mask_index(val_masks_path)
    train_names = CrackDataset.list_valid_images(
        train_images_path,
        train_masks_path,
        mask_prefix=args.mask_prefix,
        mask_index=train_mask_index,
    )
    val_names = CrackDataset.list_valid_images(
        val_images_path,
        val_masks_path,
        mask_prefix=args.mask_prefix,
        mask_index=val_mask_index,
    )
    if not train_names:
        raise RuntimeError(
            f"No valid image-mask pairs found under: {train_images_path} and {train_masks_path}"
        )
    if not val_names:
        raise RuntimeError(
            f"No valid image-mask pairs found under: {val_images_path} and {val_masks_path}"
        )

    # Resolve preprocess modes (train vs val)
    train_preprocess = getattr(args, "preprocess_train", None) or args.preprocess
    val_preprocess = getattr(args, "preprocess_val", None) or args.preprocess
    print(f"Preprocess: train={train_preprocess} | val={val_preprocess}")

    # Auto pos_weight (optional)
    pos_weight_value = args.pos_weight
    auto_requested = isinstance(pos_weight_value, str) and pos_weight_value.lower() == "auto"
    if not auto_requested:
        try:
            pos_weight_value = float(pos_weight_value)
        except (TypeError, ValueError):
            auto_requested = True
    if auto_requested or (isinstance(pos_weight_value, (int, float)) and float(pos_weight_value) <= 0):
        min_w = float(getattr(args, "pos_weight_min", 1.0) or 1.0)
        max_w = float(getattr(args, "pos_weight_max", 20.0) or 20.0)
        sample_n = int(getattr(args, "pos_weight_sample", 200) or 200)
        auto_info = _estimate_pos_weight(
            train_names=train_names,
            mask_dir=train_masks_path,
            mask_prefix=args.mask_prefix,
            mask_index=train_mask_index,
            sample_size=sample_n,
            min_weight=min_w,
            max_weight=max_w,
        )
        if auto_info is not None:
            pos_weight_value, pos_ratio, used = auto_info
            print(
                f"Auto pos_weight: {pos_weight_value:.2f} (pos_ratio={pos_ratio:.4f}, samples={used})"
            )
        else:
            pos_weight_value = 1.0
            print("Auto pos_weight failed (no valid masks). Falling back to 1.0.")
    args.pos_weight = float(pos_weight_value)

    # Preprocessing / datasets
    def _build_train_dataset(mode):
        if mode == "patch":
            image_transform = transforms.Compose([transforms.ToTensor()])
            mask_transform = transforms.Compose([transforms.ToTensor()])
            return RandomPatchDataset(
                image_dir=train_images_path,
                mask_dir=train_masks_path,
                image_filenames=train_names,
                mask_prefix=args.mask_prefix,
                mask_index=train_mask_index,
                patch_size=args.input_size,
                patches_per_image=args.patches_per_image,
                max_patch_tries=args.max_patch_tries,
                augment=not args.no_augment,
                image_transform=image_transform,
                mask_transform=mask_transform,
                verbose=True,
            )

        if mode in ["letterbox", "resize", "random_crop"]:
            return CrackDataset(
                image_dir=train_images_path,
                mask_dir=train_masks_path,
                image_filenames=train_names,
                mask_prefix=args.mask_prefix,
                mask_index=train_mask_index,
                augment=not args.no_augment,
                patch_size=None,
                output_size=args.input_size,
                verbose=True,
                cache_data=False,
                preprocess_mode=mode,
                patches_per_image=getattr(args, "patches_per_image", 1),
            )

        print(f"Warning: Unknown preprocess mode '{mode}', falling back to 'resize' for train.")
        return CrackDataset(
            image_dir=train_images_path,
            mask_dir=train_masks_path,
            image_filenames=train_names,
            mask_prefix=args.mask_prefix,
            mask_index=train_mask_index,
            augment=not args.no_augment,
            patch_size=None,
            output_size=args.input_size,
            verbose=True,
            cache_data=False,
            preprocess_mode="resize",
            patches_per_image=getattr(args, "patches_per_image", 1),
        )

    def _build_val_dataset(mode):
        if mode == "patch":
            image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            mask_transform = transforms.Compose([transforms.ToTensor()])
            stride = args.val_stride if args.val_stride and args.val_stride > 0 else args.input_size // 2
            return TiledDataset(
                image_dir=val_images_path,
                mask_dir=val_masks_path,
                image_filenames=val_names,
                mask_prefix=args.mask_prefix,
                mask_index=val_mask_index,
                patch_size=args.input_size,
                stride=stride,
                image_transform=image_transform,
                mask_transform=mask_transform,
                verbose=True,
            )

        if mode in ["letterbox", "resize", "random_crop"]:
            return CrackDataset(
                image_dir=val_images_path,
                mask_dir=val_masks_path,
                image_filenames=val_names,
                mask_prefix=args.mask_prefix,
                mask_index=val_mask_index,
                augment=False,
                patch_size=None,
                output_size=args.input_size,
                verbose=True,
                cache_data=False,
                preprocess_mode=mode,
                patches_per_image=1,
            )

        print(f"Warning: Unknown preprocess mode '{mode}', falling back to 'resize' for val.")
        return CrackDataset(
            image_dir=val_images_path,
            mask_dir=val_masks_path,
            image_filenames=val_names,
            mask_prefix=args.mask_prefix,
            mask_index=val_mask_index,
            augment=False,
            patch_size=None,
            output_size=args.input_size,
            verbose=True,
            cache_data=False,
            preprocess_mode="resize",
            patches_per_image=1,
        )

    train_dataset = _build_train_dataset(train_preprocess)
    val_dataset = _build_val_dataset(val_preprocess)

    print(f'Train set size: {len(train_dataset)}')
    print(f'Val set size: {len(val_dataset)}')

    # Data loaders.
    pin_memory = args.pin_memory
    if pin_memory is None:
        pin_memory = device.type == "cuda"
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_skip_none,
        "pin_memory": bool(pin_memory),
    }
    if device.type == "cuda" and loader_kwargs["pin_memory"]:
        loader_kwargs["pin_memory_device"] = "cuda"
    if args.num_workers > 0:
        if bool(args.persistent_workers):
            loader_kwargs["persistent_workers"] = True
        prefetch_factor = int(args.prefetch_factor) if args.prefetch_factor is not None else None
        if prefetch_factor and prefetch_factor > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    try:
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    except TypeError:
        loader_kwargs.pop("pin_memory_device", None)
        loader_kwargs.pop("persistent_workers", None)
        loader_kwargs.pop("prefetch_factor", None)
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # Model: Using Segmentation Models Pytorch (SMP) for SOTA backbones
    import segmentation_models_pytorch as smp

    # Default to a strong backbone if not specified
    encoder_name = getattr(args, "encoder_name", "efficientnet-b4")
    encoder_weights = getattr(args, "encoder_weights", "imagenet")
    classes = 1
    activation = None # We use BCEWithLogitsLoss, so no activation at output

    print(f"Model: Unet (smp) | Encoder: {encoder_name} | Weights: {encoder_weights} | Attention: SCSE")
    model = smp.Unet(
        encoder_name=encoder_name, 
        encoder_weights=encoder_weights, 
        in_channels=3, 
        classes=classes, 
        activation=activation,
        decoder_attention_type="scse" # <--- SOTA Attention for Cracks
    ).to(device)

    # Loss + optimizer.
    pos_weight = torch.tensor([float(args.pos_weight)], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def criterion(pred, target):
        loss = 0.0
        bce_weight = float(args.bce_weight)
        dice_weight = float(args.dice_weight)
        focal_weight = float(getattr(args, "focal_weight", 0.0))
        if bce_weight > 0:
            loss += bce_weight * bce(pred, target)
        if dice_weight > 0:
            loss += dice_weight * dice_loss(pred, target)
        if focal_weight > 0:
            alpha = float(getattr(args, "focal_alpha", 0.25))
            gamma = float(getattr(args, "focal_gamma", 2.0))
            loss += focal_weight * focal_loss_with_logits(pred, target, alpha=alpha, gamma=gamma)
        return loss

    # Optimizer: AdamW is generally better than Adam for SOTA models
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    # LR scheduler: CosineAnnealingWarmRestarts (SGDR) is SOTA standard
    # T_0: Number of epochs for the first restart.
    # T_mult: Factor to increase T_i after a restart.
    T_0 = int(getattr(args, "scheduler_t0", 10))
    T_mult = int(getattr(args, "scheduler_tmult", 2))
    
    print(f"Scheduler: CosineAnnealingWarmRestarts (T_0={T_0}, T_mult={T_mult})")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=float(args.learning_rate) * 0.01
    )
    
    # Allow fallback to ReduceLROnPlateau if explicitly requested (not implemented here to keep clean SOTA flow)

    if train_preprocess == "patch":
        print(
            f"Patch training: patch_size={args.input_size}, train_patches_per_image={args.patches_per_image}, "
            f"val_stride={args.val_stride if args.val_stride > 0 else (args.input_size // 2)}"
        )
    print(
        "Loss: "
        f"bce_weight={args.bce_weight}, dice_weight={args.dice_weight}, "
        f"focal_weight={getattr(args, 'focal_weight', 0.0)}, "
        f"focal_alpha={getattr(args, 'focal_alpha', 0.25)}, "
        f"focal_gamma={getattr(args, 'focal_gamma', 2.0)}, "
        f"pos_weight={args.pos_weight}"
    )
    print(f"Val metric threshold: {args.metric_threshold} | Scheduler metric: {args.scheduler_metric}")

    train_model._metric_threshold = float(args.metric_threshold)
    train_model._scheduler_metric = args.scheduler_metric
    train_model._disable_visuals = bool(args.no_visualize)
    train_model._disable_loss_curve = bool(args.no_loss_curve)
    train_model._early_stop_patience = int(args.early_stop_patience)
    train_model._visualize_every = int(args.visualize_every)
    if args.metric_thresholds.strip():
        train_model._metric_thresholds = [
            float(x.strip()) for x in args.metric_thresholds.split(",") if x.strip()
        ]
    else:
        train_model._metric_thresholds = []

    # Logging Setup
    logging.basicConfig(
        filename=os.path.join(output_dir, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )
    # Add console handler if not present (to avoid double log)
    if not any(isinstance(h, logging.StreamHandler) for h in logging.getLogger().handlers):
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    logging.info(f"Training started. Output dir: {output_dir}")
    logging.info(f"Config: {vars(args)}")

    # CSV Writer Init
    csv_path = os.path.join(output_dir, "val_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "lr"])

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,  # Pass the scheduler
        num_epochs=args.epochs,
        device=device,
        output_dir=output_dir,
        use_amp=True, # Enable FP16 Mixed Precision for speed & memory savings

        csv_path=csv_path, # Pass CSV path
        grad_accum_steps=getattr(args, "grad_accum_steps", 1)
    )
