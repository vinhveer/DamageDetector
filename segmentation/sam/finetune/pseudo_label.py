import argparse
import csv
import os
import shutil

import cv2
import numpy as np
from torch_runtime import cudnn, torch
from ..backbones.segment_anything import sam_model_registry

try:
    from .runtime import (
        apply_delta_to_sam,
        load_inference_config,
        resolve_decoder_type,
        resolve_image_size,
        resolve_predict_threshold,
        resolve_refine_settings,
        resolve_tile_settings,
    )
    from .tiled_inference import coarse_refine_model_score_map, tiled_model_score_map
except ImportError:
    from .runtime import (
        apply_delta_to_sam,
        load_inference_config,
        resolve_decoder_type,
        resolve_image_size,
        resolve_predict_threshold,
        resolve_refine_settings,
        resolve_tile_settings,
    )
    from .tiled_inference import coarse_refine_model_score_map, tiled_model_score_map


def _dataset_api():
    from ...datasets import sam_finetune as sam_datasets

    return sam_datasets


def _load_finetuned_sam(*, ckpt, vit_name, img_size, delta_type, delta_ckpt, middle_dim, scaling_factor, rank, decoder_type="auto", centerline_head=False):
    decoder = resolve_decoder_type(delta_ckpt, decoder_type)
    image_size = resolve_image_size(delta_ckpt, img_size if int(img_size) > 0 else None)
    centerline = bool(centerline_head or load_inference_config(delta_ckpt).get("centerline_head", False))
    sam, _ = sam_model_registry[vit_name](
        image_size=int(image_size),
        num_classes=1,
        checkpoint=ckpt,
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        decoder_type=decoder,
        centerline_head=centerline,
    )
    apply_delta_to_sam(
        sam=sam,
        delta_type=delta_type,
        delta_ckpt_path=delta_ckpt,
        middle_dim=int(middle_dim),
        scaling_factor=float(scaling_factor),
        rank=int(rank),
    )
    return sam.cuda(), int(image_size)


def _score_map_ensemble(image_hwc, models, image_sizes, *, tile_size, tile_overlap, tile_batch_size: int = 1):
    maps = []
    for model, image_size in zip(models, image_sizes):
        maps.append(
            tiled_model_score_map(
                image_hwc,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                model=model,
                image_size=int(image_size),
                multimask_output=False,
                use_amp=False,
                tile_batch_size=tile_batch_size,
            )
        )
    return np.mean(np.stack(maps, axis=0), axis=0).astype(np.float32)


def _resolve_image_dir(volume_path: str) -> str:
    candidate = os.path.join(volume_path, "images")
    return candidate if os.path.isdir(candidate) else volume_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pseudo-label datasets for SAM finetune self-training.")
    parser.add_argument("--volume_path", required=True, help="Dataset root containing images/ or a direct image directory")
    parser.add_argument("--output_root", required=True, help="Output dataset root with images/ and masks/")
    parser.add_argument("--ckpt", required=True, help="Base SAM checkpoint")
    parser.add_argument("--vit_name", default="vit_b")
    parser.add_argument("--delta_ckpt", required=True)
    parser.add_argument("--delta_type", required=True, choices=["adapter", "lora", "both"])
    parser.add_argument("--middle_dim", type=int, default=32)
    parser.add_argument("--scaling_factor", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--decoder_type", default="auto", choices=["auto", "baseline", "hq"])
    parser.add_argument("--centerline_head", action="store_true")
    parser.add_argument("--predict_mode", default="tile_full_box", choices=["tile_full_box", "coarse_refine"])
    parser.add_argument("--pred_threshold", default="auto")
    parser.add_argument("--tile_size", type=int, default=-1)
    parser.add_argument("--tile_overlap", type=int, default=-1)
    parser.add_argument("--tile_batch_size", type=int, default=4)
    parser.add_argument("--ensemble_delta_ckpts", nargs="*", default=None)
    parser.add_argument("--ensemble_delta_types", nargs="*", default=None)
    parser.add_argument("--ensemble_ranks", type=int, nargs="*", default=None)
    parser.add_argument("--ensemble_decoder_types", nargs="*", default=None)
    parser.add_argument("--refine_delta_ckpt", default="")
    parser.add_argument("--refine_delta_type", default="")
    parser.add_argument("--refine_rank", type=int, default=-1)
    parser.add_argument("--refine_decoder_type", default="auto", choices=["auto", "baseline", "hq"])
    parser.add_argument("--refine_centerline_head", action="store_true")
    parser.add_argument("--refine_tile_size", type=int, default=-1)
    parser.add_argument("--refine_tile_sizes", type=int, nargs="*", default=None)
    parser.add_argument("--refine_batch_size", type=int, default=2)
    parser.add_argument("--refine_max_rois", type=int, default=16)
    parser.add_argument("--refine_roi_padding", type=int, default=64)
    parser.add_argument("--refine_merge_mode", default="weighted_replace")
    parser.add_argument("--refine_score_threshold", type=float, default=0.15)
    parser.add_argument("--roi_positive_band_low", type=float, default=0.20)
    parser.add_argument("--roi_positive_band_high", type=float, default=0.90)
    parser.add_argument("--min_positive_pixels", type=int, default=64)
    parser.add_argument("--min_positive_confidence", type=float, default=0.65)
    parser.add_argument("--copy_images", action="store_true", help="Copy images into output_root/images. Otherwise symlink when possible.")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    cudnn.benchmark = False
    cudnn.deterministic = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    os.makedirs(args.output_root, exist_ok=True)
    out_img_dir = os.path.join(args.output_root, "images")
    out_mask_dir = os.path.join(args.output_root, "masks")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    coarse_model, coarse_image_size = _load_finetuned_sam(
        ckpt=args.ckpt,
        vit_name=args.vit_name,
        img_size=args.tile_size if int(args.tile_size) > 0 else 512,
        delta_type=args.delta_type,
        delta_ckpt=args.delta_ckpt,
        middle_dim=args.middle_dim,
        scaling_factor=args.scaling_factor,
        rank=args.rank,
        decoder_type=args.decoder_type,
        centerline_head=args.centerline_head,
    )
    ensemble_models = [coarse_model]
    ensemble_image_sizes = [int(coarse_image_size)]
    for idx, extra_ckpt in enumerate(list(args.ensemble_delta_ckpts or [])):
        extra_model, extra_size = _load_finetuned_sam(
            ckpt=args.ckpt,
            vit_name=args.vit_name,
            img_size=args.tile_size if int(args.tile_size) > 0 else 512,
            delta_type=(args.ensemble_delta_types or [args.delta_type])[idx] if idx < len(args.ensemble_delta_types or []) else args.delta_type,
            delta_ckpt=extra_ckpt,
            middle_dim=args.middle_dim,
            scaling_factor=args.scaling_factor,
            rank=(args.ensemble_ranks or [args.rank])[idx] if idx < len(args.ensemble_ranks or []) else args.rank,
            decoder_type=(args.ensemble_decoder_types or [args.decoder_type])[idx] if idx < len(args.ensemble_decoder_types or []) else args.decoder_type,
            centerline_head=args.centerline_head,
        )
        ensemble_models.append(extra_model)
        ensemble_image_sizes.append(int(extra_size))

    refine_model = None
    refine_image_size = None
    refine_settings = None
    if args.predict_mode == "coarse_refine":
        if not str(args.refine_delta_ckpt).strip():
            raise ValueError("coarse_refine pseudo-labeling requires --refine_delta_ckpt.")
        refine_model, refine_image_size = _load_finetuned_sam(
            ckpt=args.ckpt,
            vit_name=args.vit_name,
            img_size=args.refine_tile_size if int(args.refine_tile_size) > 0 else 768,
            delta_type=args.refine_delta_type or args.delta_type,
            delta_ckpt=args.refine_delta_ckpt,
            middle_dim=args.middle_dim,
            scaling_factor=args.scaling_factor,
            rank=args.refine_rank if int(args.refine_rank) > 0 else args.rank,
            decoder_type=args.refine_decoder_type,
            centerline_head=args.refine_centerline_head,
        )
        refine_settings = resolve_refine_settings(
            args.refine_delta_ckpt,
            refine_tile_size=args.refine_tile_size,
            refine_tile_sizes=args.refine_tile_sizes,
            refine_max_rois=args.refine_max_rois,
            refine_roi_padding=args.refine_roi_padding,
            refine_merge_mode=args.refine_merge_mode,
            refine_score_threshold=args.refine_score_threshold,
            positive_band_low=args.roi_positive_band_low,
            positive_band_high=args.roi_positive_band_high,
        )

    image_dir = _resolve_image_dir(args.volume_path)
    image_names = _dataset_api().list_image_files(image_dir)
    tile_size, tile_overlap = resolve_tile_settings(args.delta_ckpt, args.tile_size, args.tile_overlap)
    threshold = resolve_predict_threshold(args.delta_ckpt, args.pred_threshold)
    meta_path = os.path.join(args.output_root, "pseudo_label_metadata.csv")
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "positive_pixels", "positive_confidence", "saved"])
        for image_name in image_names:
            image_path = os.path.join(image_dir, image_name)
            image_hwc = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            coarse_score_map = _score_map_ensemble(
                image_hwc,
                ensemble_models,
                ensemble_image_sizes,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                tile_batch_size=int(args.tile_batch_size),
            )
            if args.predict_mode == "coarse_refine":
                score_map, _coarse_map, _refine_outputs = coarse_refine_model_score_map(
                    image_hwc,
                    coarse_model=coarse_model,
                    coarse_image_size=int(coarse_image_size),
                    coarse_tile_size=int(tile_size),
                    coarse_tile_overlap=int(tile_overlap),
                    refine_model=refine_model,
                    refine_image_size=int(refine_image_size),
                    refine_tile_size=int(refine_settings["refine_tile_size"]),
                    refine_tile_sizes=refine_settings["refine_tile_sizes"],
                    refine_max_rois=int(refine_settings["refine_max_rois"]),
                    refine_roi_padding=int(refine_settings["refine_roi_padding"]),
                    refine_merge_mode=str(refine_settings["refine_merge_mode"]),
                    refine_score_threshold=float(refine_settings["refine_score_threshold"]),
                    positive_band_low=float(refine_settings["positive_band_low"]),
                    positive_band_high=float(refine_settings["positive_band_high"]),
                    threshold=float(threshold),
                    multimask_output=False,
                    use_amp=False,
                    coarse_score_map=coarse_score_map,
                    tile_batch_size=int(args.tile_batch_size),
                    refine_batch_size=int(args.refine_batch_size),
                )
            else:
                score_map = coarse_score_map

            pred_mask = (score_map >= float(threshold)).astype(np.uint8)
            positive_pixels = int(pred_mask.sum())
            positive_conf = float(score_map[pred_mask > 0].mean()) if positive_pixels > 0 else 0.0
            saved = int(positive_pixels >= int(args.min_positive_pixels) and positive_conf >= float(args.min_positive_confidence))
            writer.writerow([image_name, positive_pixels, positive_conf, saved])
            if not saved:
                continue

            dst_img = os.path.join(out_img_dir, image_name)
            dst_mask = os.path.join(out_mask_dir, os.path.splitext(image_name)[0] + ".png")
            if args.copy_images:
                shutil.copy2(image_path, dst_img)
            else:
                if os.path.lexists(dst_img):
                    os.remove(dst_img)
                os.symlink(os.path.abspath(image_path), dst_img)
            cv2.imwrite(dst_mask, pred_mask * 255)

    print(f"Saved pseudo labels to {args.output_root}")
