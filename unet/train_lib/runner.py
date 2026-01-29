import csv
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

from unet.unet_model import UNet
from dataset_lib import (
    CrackDataset,
    LetterboxResize,
    RandomPatchDataset,
    TiledDataset,
    build_mask_index,
    find_mask_path,
)
from .collate import collate_skip_none
from .losses import dice_loss
from .training import train_model


def _cache_root(args):
    if not args.cache_dir:
        return None
    if args.preprocess not in ("letterbox", "stretch"):
        print("Cache disabled: only supported for --preprocess letterbox/stretch.")
        return None
    if not args.no_augment:
        print("Cache disabled: requires --no-augment to keep data identical.")
        return None
    return os.path.join(args.cache_dir, f"{args.preprocess}_{args.input_size}")


def _cache_transforms(preprocess, input_size):
    if preprocess == "letterbox":
        image_tf = LetterboxResize(input_size, fill=(0, 0, 0), interpolation=Image.BILINEAR)
        mask_tf = LetterboxResize(input_size, fill=0, interpolation=Image.NEAREST)
        return image_tf, mask_tf
    if preprocess == "stretch":
        image_tf = transforms.Resize(
            (input_size, input_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask_tf = transforms.Resize(
            (input_size, input_size),
            interpolation=InterpolationMode.NEAREST,
            antialias=False,
        )
        return image_tf, mask_tf
    raise ValueError(f"Cache preprocess not supported: {preprocess}")


def _prepare_cache_split(
    split_name,
    image_dir,
    mask_dir,
    image_filenames,
    mask_prefix,
    mask_index,
    cache_root,
    preprocess,
    input_size,
    rebuild,
):
    image_tf, mask_tf = _cache_transforms(preprocess, input_size)
    split_root = os.path.join(cache_root, split_name)
    images_out = os.path.join(split_root, "images")
    masks_out = os.path.join(split_root, "masks")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(masks_out, exist_ok=True)

    manifest_path = os.path.join(split_root, "manifest.csv")
    cached_names = []
    created = 0
    reused = 0
    skipped = 0

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_src", "mask_src", "image_cache", "mask_cache"])

        for img_name in image_filenames:
            base_name = os.path.splitext(img_name)[0]
            mask_path = find_mask_path(
                mask_dir,
                base_name,
                mask_prefix=mask_prefix,
                mask_index=mask_index,
            )
            if mask_path is None:
                skipped += 1
                continue

            image_src = os.path.join(image_dir, img_name)
            mask_src = mask_path
            image_cache_name = f"{base_name}.png"
            mask_stem = os.path.splitext(os.path.basename(mask_path))[0]
            mask_cache_name = f"{mask_stem}.png"
            image_cache = os.path.join(images_out, image_cache_name)
            mask_cache = os.path.join(masks_out, mask_cache_name)

            need_write = rebuild or not (os.path.exists(image_cache) and os.path.exists(mask_cache))
            if need_write:
                try:
                    with Image.open(image_src) as img:
                        img = img.convert("RGB")
                        img = image_tf(img)
                        img.save(image_cache)
                    with Image.open(mask_src) as m:
                        m = m.convert("L")
                        m = mask_tf(m)
                        m.save(mask_cache)
                    created += 1
                except Exception as exc:
                    print(f"Cache warning: failed to process {img_name}: {exc}")
                    skipped += 1
                    continue
            else:
                reused += 1

            cached_names.append(image_cache_name)
            writer.writerow(
                [
                    os.path.abspath(image_src),
                    os.path.abspath(mask_src),
                    os.path.abspath(image_cache),
                    os.path.abspath(mask_cache),
                ]
            )

    print(
        f"Cache {split_name}: {len(cached_names)} pair(s) "
        f"(created={created}, reused={reused}, skipped={skipped}) -> {split_root}"
    )
    return images_out, masks_out, cached_names


def run_training(args):
    # Select device.
    device = torch.device("cpu")
    print(f'Using device: {device}')

    # Output directory.
    output_dir = args.output_dir
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

    cache_root = _cache_root(args)
    cache_enabled = cache_root is not None
    if cache_enabled:
        os.makedirs(cache_root, exist_ok=True)
        train_images_path, train_masks_path, train_names = _prepare_cache_split(
            "train",
            train_images_path,
            train_masks_path,
            train_names,
            args.mask_prefix,
            train_mask_index,
            cache_root,
            args.preprocess,
            args.input_size,
            args.cache_rebuild,
        )
        val_images_path, val_masks_path, val_names = _prepare_cache_split(
            "val",
            val_images_path,
            val_masks_path,
            val_names,
            args.mask_prefix,
            val_mask_index,
            cache_root,
            args.preprocess,
            args.input_size,
            args.cache_rebuild,
        )
        train_mask_index = build_mask_index(train_masks_path)
        val_mask_index = build_mask_index(val_masks_path)
        if not train_names:
            raise RuntimeError(f"No cached train pairs found under: {train_images_path} and {train_masks_path}")
        if not val_names:
            raise RuntimeError(f"No cached val pairs found under: {val_images_path} and {val_masks_path}")

    # Preprocessing / datasets
    if args.preprocess == "patch":
        image_transform = transforms.Compose([transforms.ToTensor()])
        mask_transform = transforms.Compose([transforms.ToTensor()])

        train_dataset = RandomPatchDataset(
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

        stride = args.val_stride if args.val_stride and args.val_stride > 0 else args.input_size // 2
        val_dataset = TiledDataset(
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

    elif args.preprocess == "letterbox":
        # Keep aspect ratio: resize to fit, then pad to square.
        if cache_enabled:
            image_transform = transforms.Compose([transforms.ToTensor()])
            mask_transform = transforms.Compose([transforms.ToTensor()])
        else:
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
            mask_index=train_mask_index,
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
            mask_index=val_mask_index,
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
        if cache_enabled:
            image_transform = transforms.Compose([transforms.ToTensor()])
            mask_transform = transforms.Compose([transforms.ToTensor()])
        else:
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
            mask_index=train_mask_index,
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
            mask_index=val_mask_index,
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

    # Model.
    model = UNet(in_channels=3, out_channels=1).to(device)

    # Loss + optimizer.
    pos_weight = torch.tensor([float(args.pos_weight)], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def criterion(pred, target):
        return float(args.bce_weight) * bce(pred, target) + float(args.dice_weight) * dice_loss(pred, target)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    # LR scheduler.
    sched_mode = "max" if args.scheduler_metric == "dice" else "min"
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_mode,
            factor=float(args.scheduler_factor),
            patience=int(args.scheduler_patience),
            verbose=True,
        )
    except TypeError:
        # Newer PyTorch versions removed the 'verbose' argument.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_mode,
            factor=float(args.scheduler_factor),
            patience=int(args.scheduler_patience),
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
