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
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install PyYAML")

    # Load model config from YAML
    model_config_path = getattr(args, "model_config", "unet/model_config.yaml")
    if not os.path.exists(model_config_path):
        # Fallback to current dir
        if os.path.exists("model_config.yaml"):
            model_config_path = "model_config.yaml"
        else:
             print(f"Warning: Model config not found at {model_config_path}. Using defaults.")
             model_config = {}

    if os.path.exists(model_config_path):
        with open(model_config_path, "r") as f:
            full_config = yaml.safe_load(f) or {}
            model_config = full_config.get("model", full_config) # Handle nested or flat structure
            print(f"Loaded model config from {model_config_path}: {model_config}")
    
    # Merge defaults from model_config into args
    training_cfg = model_config.get("training", {})
    dataloader_cfg = model_config.get("dataloader", {})
    model_cfg = model_config.get("model", model_config) # fallback if flat

    # Helper to prioritize YAML if default, otherwise keep CLI arg
    def override_if_default(arg_val, yaml_key, default_val, section=training_cfg):
        if arg_val == default_val and yaml_key in section:
             return section[yaml_key]
        return arg_val

    # Preprocessing
    args.preprocess = override_if_default(args.preprocess, "preprocess", "patch")
    args.preprocess_train = override_if_default(args.preprocess_train, "preprocess_train", None)
    args.preprocess_val = override_if_default(args.preprocess_val, "preprocess_val", None)
    args.input_size = override_if_default(args.input_size, "input_size", 256)
    args.patches_per_image = override_if_default(args.patches_per_image, "patches_per_image", 1)
    args.max_patch_tries = override_if_default(args.max_patch_tries, "max_patch_tries", 5)
    args.val_stride = override_if_default(args.val_stride, "val_stride", 0)
    args.mask_prefix = override_if_default(args.mask_prefix, "mask_prefix", "auto")

    # Augmentation
    if "no_augment" in training_cfg and training_cfg["no_augment"]:
         args.no_augment = True
    args.aug_prob = override_if_default(args.aug_prob, "aug_prob", 0.5)
    args.rotate_limit = override_if_default(args.rotate_limit, "rotate_limit", 10)
    args.brightness_limit = override_if_default(args.brightness_limit, "brightness_limit", 0.2)
    args.contrast_limit = override_if_default(args.contrast_limit, "contrast_limit", 0.2)
    
    # Caching
    if "cache_data" in training_cfg and training_cfg["cache_data"]:
         args.cache_data = True

    # Optimization
    args.learning_rate = override_if_default(args.learning_rate, "learning_rate", 0.0005)
    args.weight_decay = override_if_default(args.weight_decay, "weight_decay", 1e-5)
    args.grad_accum_steps = override_if_default(getattr(args, "grad_accum_steps", 1), "grad_accum_steps", 1)
    args.early_stop_patience = override_if_default(getattr(args, "early_stop_patience", 15), "early_stop_patience", 15)

    # Scheduler
    args.scheduler_t0 = override_if_default(getattr(args, "scheduler_t0", 10), "scheduler_t0", 10)
    args.scheduler_tmult = override_if_default(getattr(args, "scheduler_tmult", 2), "scheduler_tmult", 2)
    args.scheduler_metric = override_if_default(getattr(args, "scheduler_metric", "loss"), "scheduler_metric", "loss")

    # Loss Weights
    if args.pos_weight == "5.0" and "pos_weight" in training_cfg:
         args.pos_weight = training_cfg["pos_weight"]

    args.bce_weight = override_if_default(args.bce_weight, "bce_weight", 0.4)
    args.dice_weight = override_if_default(args.dice_weight, "dice_weight", 0.6)
    args.focal_weight = override_if_default(getattr(args, "focal_weight", 0.0), "focal_weight", 0.0)
    args.focal_alpha = override_if_default(getattr(args, "focal_alpha", 0.25), "focal_alpha", 0.25)
    args.focal_gamma = override_if_default(getattr(args, "focal_gamma", 2.0), "focal_gamma", 2.0)
    
    # Metrics
    args.metric_threshold = override_if_default(args.metric_threshold, "metric_threshold", 0.5)
    if "metric_thresholds" in training_cfg:
         val = training_cfg["metric_thresholds"]
         if isinstance(val, list):
             val = ",".join(str(v) for v in val)
         if not args.metric_thresholds:
             args.metric_thresholds = val

    # Dataloader
    args.num_workers = override_if_default(args.num_workers, "num_workers", 8, dataloader_cfg)
    args.prefetch_factor = override_if_default(getattr(args, "prefetch_factor", 2), "prefetch_factor", 2, dataloader_cfg)
    if "pin_memory" in dataloader_cfg and not args.pin_memory:
         args.pin_memory = dataloader_cfg["pin_memory"]
    if "persistent_workers" in dataloader_cfg:
         args.persistent_workers = dataloader_cfg["persistent_workers"]

    # Model Params
    encoder_name = model_cfg.get("encoder_name", "efficientnet-b4")
    encoder_weights = model_cfg.get("encoder_weights", "imagenet")
    classes = model_cfg.get("classes", 1)
    activation = model_cfg.get("activation", None)
    decoder_attention_type = model_cfg.get("decoder_attention_type", "scse")
    
    # Init Model
    print(f"Model: Unet (smp) | Encoder: {encoder_name} | Weights: {encoder_weights} | Attention: {decoder_attention_type}")
    model = smp.Unet(
        encoder_name=encoder_name, 
        encoder_weights=encoder_weights, 
        in_channels=3, 
        classes=classes, 
        activation=activation,
        decoder_attention_type=decoder_attention_type
    ).to(device)

    # Resolve preprocess modes (train vs val) - Re-resolve after merge
    train_preprocess = getattr(args, "preprocess_train", None) or args.preprocess
    val_preprocess = getattr(args, "preprocess_val", None) or args.preprocess

    # Multi-GPU support (DataParallel)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)

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
        use_amp=False, # Disable AMP to fix CUDA misaligned address error with ConvNext/SCSE

        csv_path=csv_path, # Pass CSV path
        grad_accum_steps=getattr(args, "grad_accum_steps", 1)
    )
