import argparse
import logging
import os
import random
import sys

import numpy as np
from PIL import Image
from torch_runtime import DataLoader
from torch_runtime import cudnn, torch
from segment_anything import sam_model_registry

try:
    from datasets.dataset_generic import GenericDataset, ValGenerator, list_image_files, load_image_mask_arrays
    from runtime import (
        apply_delta_to_sam,
        load_inference_config,
        resolve_decoder_type,
        resolve_image_size,
        resolve_predict_mode,
        resolve_predict_threshold,
        resolve_refine_settings,
        resolve_tile_settings,
    )
    from tiled_inference import (
        best_threshold_result,
        coarse_refine_model_score_map,
        continuity_metrics,
        tiled_model_score_map,
        threshold_sweep,
    )
    from utils import test_single_volume
except ImportError:
    from .datasets.dataset_generic import GenericDataset, ValGenerator, list_image_files, load_image_mask_arrays
    from .runtime import (
        apply_delta_to_sam,
        load_inference_config,
        resolve_decoder_type,
        resolve_image_size,
        resolve_predict_mode,
        resolve_predict_threshold,
        resolve_refine_settings,
        resolve_tile_settings,
    )
    from .tiled_inference import (
        best_threshold_result,
        coarse_refine_model_score_map,
        continuity_metrics,
        tiled_model_score_map,
        threshold_sweep,
    )
    from .utils import test_single_volume


NUM_CLASSES = 1


def config_to_dict(config):
    items_dict = {}
    with open(config, "r", encoding="utf-8") as f:
        items = f.readlines()
    for line in items:
        if ": " not in line:
            continue
        key, value = line.strip().split(": ", 1)
        items_dict[key] = value
    return items_dict


def _load_finetuned_sam(
    *,
    ckpt: str,
    vit_name: str,
    img_size: int,
    delta_type: str,
    delta_ckpt: str,
    middle_dim: int,
    scaling_factor: float,
    rank: int,
    decoder_type: str = "auto",
    centerline_head: bool = False,
):
    decoder = resolve_decoder_type(delta_ckpt, decoder_type)
    image_size = resolve_image_size(delta_ckpt, img_size if int(img_size) > 0 else None)
    sam, _ = sam_model_registry[vit_name](
        image_size=int(image_size),
        num_classes=NUM_CLASSES,
        checkpoint=ckpt,
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        decoder_type=decoder,
        centerline_head=bool(centerline_head),
    )
    apply_delta_to_sam(
        sam=sam,
        delta_type=delta_type,
        delta_ckpt_path=delta_ckpt,
        middle_dim=int(middle_dim),
        scaling_factor=float(scaling_factor),
        rank=int(rank),
    )
    return sam.cuda(), int(image_size), decoder


def _tiled_score_map_ensemble(
    image_hwc: np.ndarray,
    models: list,
    image_sizes: list[int],
    *,
    tile_size: int,
    tile_overlap: int,
    multimask_output: bool,
    tile_batch_size: int = 1,
) -> np.ndarray:
    score_maps = []
    for model, image_size in zip(models, image_sizes):
        score_maps.append(
            tiled_model_score_map(
                image_hwc,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                model=model,
                image_size=int(image_size),
                multimask_output=multimask_output,
                use_amp=False,
                tile_batch_size=tile_batch_size,
            )
        )
    return np.mean(np.stack(score_maps, axis=0), axis=0).astype(np.float32)


def _save_case_outputs(test_save_path, case_name: str, image_hwc: np.ndarray, prob_map: np.ndarray, label: np.ndarray, threshold: float) -> None:
    pred = (prob_map >= float(threshold)).astype(np.uint8) * 255
    image_uint8 = image_hwc
    if image_uint8.dtype != np.uint8:
        if image_uint8.max() <= 1.0:
            image_uint8 = np.clip(image_uint8 * 255.0, 0, 255).astype(np.uint8)
        else:
            image_uint8 = np.clip(image_uint8, 0, 255).astype(np.uint8)

    Image.fromarray(image_uint8).save(os.path.join(test_save_path, "img", case_name + "_img.jpg"))
    Image.fromarray(pred).save(os.path.join(test_save_path, "pred", case_name + "_img.jpg"))
    Image.fromarray((label.astype(np.uint8) * 255)).save(os.path.join(test_save_path, "gt", case_name + "_img.jpg"))


def _run_tiled_eval(args, model, multimask_output, test_save_path=None, *, ensemble_models=None, ensemble_image_sizes=None):
    case_names = list_image_files(os.path.join(args.volume_path, "images"))
    logging.info("%d tiled full-image test iterations", len(case_names))

    metric_by_thr = {float(thr): np.zeros(4, dtype=np.float64) for thr in args.val_thresholds}
    continuity_by_thr = {
        float(thr): {
            "skeleton_dice": 0.0,
            "centerline_precision": 0.0,
            "centerline_recall": 0.0,
            "component_fragmentation": 0.0,
        }
        for thr in args.val_thresholds
    }
    total_cases = 0
    saved_cases = []
    tile_size, tile_overlap = resolve_tile_settings(args.delta_ckpt, args.tile_size, args.tile_overlap)

    for i_batch, case_name in enumerate(case_names):
        image_hwc, label = load_image_mask_arrays(args.volume_path, case_name)
        if ensemble_models:
            score_map = _tiled_score_map_ensemble(
                image_hwc,
                ensemble_models,
                ensemble_image_sizes,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                multimask_output=multimask_output,
                tile_batch_size=int(getattr(args, "tile_batch_size", 1)),
            )
        else:
            score_map = tiled_model_score_map(
                image_hwc,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                model=model,
                image_size=int(args.img_size),
                multimask_output=multimask_output,
                use_amp=False,
                tile_batch_size=int(getattr(args, "tile_batch_size", 1)),
            )
        sweep = threshold_sweep(score_map, label, args.val_thresholds)
        total_cases += 1
        case_logs = []
        for thr in args.val_thresholds:
            metric_by_thr[float(thr)] += np.array(sweep[float(thr)], dtype=np.float64)
            continuity = continuity_metrics((score_map >= float(thr)).astype(np.uint8), label)
            for key in continuity_by_thr[float(thr)]:
                continuity_by_thr[float(thr)][key] += float(continuity[key])
            case_logs.append(f"thr={float(thr):.2f} tile_iou={sweep[float(thr)][3]:.4f}")
        logging.info("idx %d case %s %s", i_batch, case_name, " | ".join(case_logs))
        if test_save_path is not None:
            saved_cases.append((case_name, image_hwc, score_map, label))

    metric_by_thr = {
        float(thr): metric_by_thr[float(thr)] / max(1, total_cases)
        for thr in args.val_thresholds
    }
    best_thr, best_metric = best_threshold_result(metric_by_thr)
    save_threshold = resolve_predict_threshold(args.delta_ckpt, args.pred_threshold)
    if str(args.pred_threshold).strip().lower() == "auto":
        save_threshold = float(best_thr)

    continuity_totals = {
        key: float(value) / max(1, total_cases)
        for key, value in continuity_by_thr[float(best_thr)].items()
    }
    if test_save_path is not None:
        for case_name, image_hwc, prob_map, label in saved_cases:
            _save_case_outputs(test_save_path, case_name, image_hwc, prob_map, label, save_threshold)

    logging.info(
        "Testing tile_full_box: mean_pr %f mean_re %f mean_f1 %f mean_iou : %f",
        best_metric[0],
        best_metric[1],
        best_metric[2],
        best_metric[3],
    )
    logging.info("Testing tile_full_box selected threshold: %.2f", float(best_thr))
    logging.info(
        "Testing tile_full_box continuity @ %.2f: skeleton_dice=%f centerline_precision=%f centerline_recall=%f component_fragmentation=%f",
        float(best_thr),
        float(continuity_totals["skeleton_dice"]),
        float(continuity_totals["centerline_precision"]),
        float(continuity_totals["centerline_recall"]),
        float(continuity_totals["component_fragmentation"]),
    )
    return {
        "best_threshold": float(best_thr),
        "save_threshold": float(save_threshold),
        "best_metric": best_metric,
        "metric_by_thr": metric_by_thr,
        "continuity": continuity_totals,
    }


def _run_coarse_refine_eval(
    args,
    coarse_model,
    refine_model,
    multimask_output,
    test_save_path=None,
    *,
    ensemble_models=None,
    ensemble_image_sizes=None,
):
    case_names = list_image_files(os.path.join(args.volume_path, "images"))
    logging.info("%d coarse_refine full-image test iterations", len(case_names))

    metric_by_thr = {float(thr): np.zeros(4, dtype=np.float64) for thr in args.val_thresholds}
    continuity_by_thr = {
        float(thr): {
            "skeleton_dice": 0.0,
            "centerline_precision": 0.0,
            "centerline_recall": 0.0,
            "component_fragmentation": 0.0,
        }
        for thr in args.val_thresholds
    }
    total_cases = 0
    saved_cases = []
    tile_size, tile_overlap = resolve_tile_settings(args.delta_ckpt, args.tile_size, args.tile_overlap)
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
    coarse_image_size = resolve_image_size(args.delta_ckpt, args.img_size)
    refine_image_size = resolve_image_size(args.refine_delta_ckpt, args.refine_tile_size)

    for i_batch, case_name in enumerate(case_names):
        image_hwc, label = load_image_mask_arrays(args.volume_path, case_name)
        if ensemble_models:
            coarse_score_map = _tiled_score_map_ensemble(
                image_hwc,
                ensemble_models,
                ensemble_image_sizes,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                multimask_output=multimask_output,
                tile_batch_size=int(getattr(args, "tile_batch_size", 1)),
            )
        else:
            coarse_score_map = None
        score_map, coarse_map, refine_outputs = coarse_refine_model_score_map(
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
            threshold=float(resolve_predict_threshold(args.delta_ckpt, args.pred_threshold)),
            multimask_output=multimask_output,
            use_amp=False,
            coarse_score_map=coarse_score_map,
            tile_batch_size=int(getattr(args, "tile_batch_size", 1)),
            refine_batch_size=int(getattr(args, "refine_batch_size", getattr(args, "tile_batch_size", 1))),
        )
        sweep = threshold_sweep(score_map, label, args.val_thresholds)
        total_cases += 1
        case_logs = []
        for thr in args.val_thresholds:
            metric_by_thr[float(thr)] += np.array(sweep[float(thr)], dtype=np.float64)
            continuity = continuity_metrics((score_map >= float(thr)).astype(np.uint8), label)
            for key in continuity_by_thr[float(thr)]:
                continuity_by_thr[float(thr)][key] += float(continuity[key])
            case_logs.append(f"thr={float(thr):.2f} coarse_refine_iou={sweep[float(thr)][3]:.4f}")
        logging.info(
            "idx %d case %s rois=%d %s",
            i_batch,
            case_name,
            len(refine_outputs),
            " | ".join(case_logs),
        )
        if test_save_path is not None:
            saved_cases.append((case_name, image_hwc, score_map, label))

    metric_by_thr = {
        float(thr): metric_by_thr[float(thr)] / max(1, total_cases)
        for thr in args.val_thresholds
    }
    best_thr, best_metric = best_threshold_result(metric_by_thr)
    save_threshold = resolve_predict_threshold(args.delta_ckpt, args.pred_threshold)
    if str(args.pred_threshold).strip().lower() == "auto":
        save_threshold = float(best_thr)

    continuity_totals = {
        key: float(value) / max(1, total_cases)
        for key, value in continuity_by_thr[float(best_thr)].items()
    }
    if test_save_path is not None:
        for case_name, image_hwc, prob_map, label in saved_cases:
            _save_case_outputs(test_save_path, case_name, image_hwc, prob_map, label, save_threshold)
    logging.info(
        "Testing coarse_refine: mean_pr %f mean_re %f mean_f1 %f mean_iou : %f",
        best_metric[0],
        best_metric[1],
        best_metric[2],
        best_metric[3],
    )
    logging.info("Testing coarse_refine selected threshold: %.2f", float(best_thr))
    logging.info(
        "Testing coarse_refine continuity @ %.2f: skeleton_dice=%f centerline_precision=%f centerline_recall=%f component_fragmentation=%f",
        float(best_thr),
        float(continuity_totals["skeleton_dice"]),
        float(continuity_totals["centerline_precision"]),
        float(continuity_totals["centerline_recall"]),
        float(continuity_totals["component_fragmentation"]),
    )
    return {
        "best_threshold": float(best_thr),
        "save_threshold": float(save_threshold),
        "best_metric": best_metric,
        "metric_by_thr": metric_by_thr,
        "continuity": continuity_totals,
    }


def _run_legacy_eval(args, model, multimask_output, test_save_path=None):
    db_test = GenericDataset(
        base_dir=args.volume_path,
        split="test_vol",
        transform=ValGenerator(output_size=[args.img_size, args.img_size], low_res=[args.img_size, args.img_size]),
    )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info("%d legacy full-box test iterations", len(testloader))

    metric_by_thr = {float(thr): np.zeros(4, dtype=np.float64) for thr in args.val_thresholds}
    total_cases = 0
    save_threshold = resolve_predict_threshold(args.delta_ckpt, args.pred_threshold)

    for i_batch, sampled_batch in enumerate(testloader):
        image = sampled_batch["image"]
        label = sampled_batch["label"]
        case_name = sampled_batch["case_name"][0]
        box = None if bool(getattr(args, "full_image_eval", False)) else sampled_batch["box"].cuda()
        case_logs = []
        for thr in args.val_thresholds:
            metric_box_i = test_single_volume(
                image,
                label,
                model,
                classes=NUM_CLASSES,
                multimask_output=multimask_output,
                patch_size=[args.img_size, args.img_size],
                test_save_path=test_save_path if float(thr) == float(save_threshold) else None,
                case=case_name,
                boxes=box,
                points=None,
                threshold_prob=float(thr),
                use_full_image_box_prompt=bool(getattr(args, "full_image_eval", False)),
            )
            metric_by_thr[float(thr)] += np.array(metric_box_i, dtype=np.float64)
            case_logs.append(f"thr={float(thr):.2f} legacy_iou={metric_box_i[3]:.4f}")
        total_cases += 1
        logging.info("legacy idx %d case %s %s", i_batch, case_name, " | ".join(case_logs))

    metric_by_thr = {
        float(thr): metric_by_thr[float(thr)] / max(1, total_cases)
        for thr in args.val_thresholds
    }
    best_thr, best_metric = best_threshold_result(metric_by_thr)
    logging.info(
        "Testing legacy_full_box: mean_pr %f mean_re %f mean_f1 %f mean_iou : %f",
        best_metric[0],
        best_metric[1],
        best_metric[2],
        best_metric[3],
    )
    logging.info("Testing legacy_full_box selected threshold: %.2f", float(best_thr))
    return {
        "best_threshold": float(best_thr),
        "save_threshold": float(save_threshold),
        "best_metric": best_metric,
        "metric_by_thr": metric_by_thr,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SAM finetune for crack segmentation using tiled full-image inference.")
    parser.add_argument("--config", type=str, default=None, help="The config file provided by the trained model")
    parser.add_argument("--volume_path", type=str, required=True, help="Dataset root containing images/ and masks/")
    parser.add_argument("--output_dir", type=str, default="./output/test")
    parser.add_argument("--img_size", type=int, default=448, help="Input image size of the network")
    parser.add_argument("--seed", type=int, default=3407, help="random seed")
    parser.add_argument("--is_savenii", action="store_true", help="Whether to save results during inference")
    parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
    parser.add_argument("--ckpt", type=str, default="checkpoints/sam_vit_h_4b8939.pth", help="Pretrained checkpoint")
    parser.add_argument("--delta_ckpt", type=str, default=None, help="The checkpoint from adapter/LoRA")
    parser.add_argument("--vit_name", type=str, default="vit_h", help="Select one vit model")
    parser.add_argument("--delta_type", type=str, default="adapter", help='choose from "adapter" or "lora" or "both"')
    parser.add_argument("--middle_dim", type=int, default=32, help="Middle dim of adapter")
    parser.add_argument("--scaling_factor", type=float, default=0.1, help="Scaling_factor of adapter")
    parser.add_argument("--rank", type=int, default=4, help="Rank for LoRA adaptation")
    parser.add_argument("--decoder_type", type=str, default="baseline", choices=["baseline", "hq"], help="Mask decoder type")
    parser.add_argument("--centerline_head", action="store_true", help="Enable dedicated centerline head when building the SAM decoder")
    parser.add_argument("--pred_threshold", default="auto", help="Probability threshold as float or 'auto'")
    parser.add_argument("--val_thresholds", type=float, nargs="+", default=[0.35, 0.4, 0.45, 0.5, 0.55, 0.6], help="Candidate probability thresholds")
    parser.add_argument("--eval_mode", default="auto", choices=["auto", "tile_full_box", "legacy_full_box", "coarse_refine"], help="Primary evaluation mode")
    parser.add_argument("--tile_size", type=int, default=-1, help="Tile size for tiled evaluation (-1 = use checkpoint metadata or 512)")
    parser.add_argument("--tile_overlap", type=int, default=-1, help="Tile overlap for tiled evaluation (-1 = use checkpoint metadata or tile_size // 2)")
    parser.add_argument("--tile_batch_size", type=int, default=4, help="Batch size for tiled full-image inference")
    parser.add_argument("--refine_delta_ckpt", type=str, default="", help="Checkpoint for the refine SAM finetune stage")
    parser.add_argument("--refine_delta_type", type=str, default="", help='Refine delta type: "adapter", "lora", or "both"')
    parser.add_argument("--refine_rank", type=int, default=-1, help="Rank for the refine LoRA adaptation")
    parser.add_argument("--refine_decoder_type", type=str, default="auto", choices=["auto", "baseline", "hq"], help="Mask decoder type for refine stage")
    parser.add_argument("--refine_centerline_head", action="store_true", help="Enable dedicated centerline head for refine decoder")
    parser.add_argument("--refine_tile_size", type=int, default=-1, help="Fixed ROI size for high-resolution refine stage")
    parser.add_argument("--refine_tile_sizes", type=int, nargs="*", default=None, help="Optional multi-scale ROI sizes for sequential refine passes")
    parser.add_argument("--refine_batch_size", type=int, default=2, help="Batch size for refine ROI inference")
    parser.add_argument("--refine_max_rois", type=int, default=16, help="Maximum number of refine ROIs per image")
    parser.add_argument("--refine_roi_padding", type=int, default=64, help="Padding around mined refine ROIs")
    parser.add_argument("--refine_merge_mode", type=str, default="weighted_replace", help="Merge mode for coarse/refine score maps")
    parser.add_argument("--refine_score_threshold", type=float, default=0.15, help="Candidate ROI heat threshold")
    parser.add_argument("--roi_positive_band_low", type=float, default=0.20, help="Low end of coarse-score band for ROI mining")
    parser.add_argument("--roi_positive_band_high", type=float, default=0.90, help="High end of coarse-score band for ROI mining")
    parser.add_argument("--ensemble_delta_ckpts", type=str, nargs="*", default=None, help="Additional coarse checkpoints for probability-map ensembling")
    parser.add_argument("--ensemble_delta_types", type=str, nargs="*", default=None, help="Delta types for ensemble checkpoints; defaults to primary delta_type")
    parser.add_argument("--ensemble_ranks", type=int, nargs="*", default=None, help="LoRA ranks for ensemble checkpoints; defaults to primary rank")
    parser.add_argument("--ensemble_decoder_types", type=str, nargs="*", default=None, help="Decoder types for ensemble checkpoints; defaults to primary decoder type")
    parser.add_argument("--legacy_box_eval", action="store_true", help="Also run legacy box-only diagnostics")
    parser.add_argument("--full_image_eval", action="store_true", help="Legacy mode: use full-image box prompt instead of dataset box")

    args = parser.parse_args()

    if args.config is not None:
        config_dict = config_to_dict(args.config)
        for key, value in config_dict.items():
            if not hasattr(args, key):
                continue
            current = getattr(args, key)
            if isinstance(current, bool):
                setattr(args, key, str(value).lower() in {"1", "true", "yes"})
            elif isinstance(current, int):
                setattr(args, key, int(value))
            elif isinstance(current, float):
                setattr(args, key, float(value))
            else:
                setattr(args, key, value)

    if not args.delta_ckpt and args.config is not None:
        config_dir = os.path.dirname(os.path.abspath(args.config))
        best_ckpt = os.path.join(config_dir, "best_model.pth")
        if os.path.isfile(best_ckpt):
            args.delta_ckpt = best_ckpt

    if not args.delta_ckpt:
        raise ValueError("delta_ckpt is required. Pass it directly or provide it via --config.")

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    sam, coarse_img_size, coarse_decoder = _load_finetuned_sam(
        ckpt=args.ckpt,
        vit_name=args.vit_name,
        img_size=args.img_size,
        delta_type=args.delta_type,
        delta_ckpt=args.delta_ckpt,
        middle_dim=args.middle_dim,
        scaling_factor=args.scaling_factor,
        rank=args.rank,
        decoder_type=args.decoder_type,
        centerline_head=bool(getattr(args, "centerline_head", False) or load_inference_config(args.delta_ckpt).get("centerline_head", False)),
    )
    args.img_size = int(coarse_img_size)
    args.decoder_type = coarse_decoder
    ensemble_models = [sam]
    ensemble_image_sizes = [int(coarse_img_size)]
    extra_ckpts = list(getattr(args, "ensemble_delta_ckpts", None) or [])
    extra_types = list(getattr(args, "ensemble_delta_types", None) or [])
    extra_ranks = list(getattr(args, "ensemble_ranks", None) or [])
    extra_decoders = list(getattr(args, "ensemble_decoder_types", None) or [])
    for idx, extra_ckpt in enumerate(extra_ckpts):
        extra_delta_type = extra_types[idx] if idx < len(extra_types) else args.delta_type
        extra_rank = extra_ranks[idx] if idx < len(extra_ranks) else args.rank
        extra_decoder = extra_decoders[idx] if idx < len(extra_decoders) else args.decoder_type
        extra_model, extra_image_size, _extra_decoder = _load_finetuned_sam(
            ckpt=args.ckpt,
            vit_name=args.vit_name,
            img_size=args.img_size,
            delta_type=extra_delta_type,
            delta_ckpt=extra_ckpt,
            middle_dim=args.middle_dim,
            scaling_factor=args.scaling_factor,
            rank=extra_rank,
            decoder_type=extra_decoder,
            centerline_head=bool(getattr(args, "centerline_head", False) or load_inference_config(extra_ckpt).get("centerline_head", False)),
        )
        ensemble_models.append(extra_model)
        ensemble_image_sizes.append(int(extra_image_size))
    refine_sam = None
    if resolved_mode == "coarse_refine":
        if not str(args.refine_delta_ckpt or "").strip():
            raise ValueError("coarse_refine evaluation requires --refine_delta_ckpt.")
        refine_delta_type = str(args.refine_delta_type or args.delta_type).strip().lower()
        refine_rank = int(args.refine_rank if int(args.refine_rank) > 0 else args.rank)
        refine_sam, _refine_img_size, _refine_decoder = _load_finetuned_sam(
            ckpt=args.ckpt,
            vit_name=args.vit_name,
            img_size=args.refine_tile_size if int(args.refine_tile_size) > 0 else args.img_size,
            delta_type=refine_delta_type,
            delta_ckpt=args.refine_delta_ckpt,
            middle_dim=args.middle_dim,
            scaling_factor=args.scaling_factor,
            rank=refine_rank,
            decoder_type=args.refine_decoder_type,
            centerline_head=bool(getattr(args, "refine_centerline_head", False) or load_inference_config(args.refine_delta_ckpt).get("centerline_head", False)),
        )
    multimask_output = False

    log_folder = os.path.join(args.output_dir, "test_log")
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_folder, "log.txt"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    args.val_thresholds = sorted(set(float(x) for x in args.val_thresholds if 0.0 < float(x) < 1.0)) or [0.5]
    resolved_mode = resolve_predict_mode(args.delta_ckpt, args.eval_mode)
    args.pred_threshold = str(args.pred_threshold)
    logging.info(str(args))
    logging.info("Resolved eval mode: %s", resolved_mode)

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, "predictions")
        os.makedirs(test_save_path, exist_ok=True)
        os.makedirs(test_save_path + "/img/", exist_ok=True)
        os.makedirs(test_save_path + "/pred/", exist_ok=True)
        os.makedirs(test_save_path + "/gt/", exist_ok=True)
    else:
        test_save_path = None

    if resolved_mode == "legacy_full_box":
        primary = _run_legacy_eval(args, sam, multimask_output, test_save_path)
    elif resolved_mode == "coarse_refine":
        primary = _run_coarse_refine_eval(
            args,
            sam,
            refine_sam,
            multimask_output,
            test_save_path,
            ensemble_models=ensemble_models if len(ensemble_models) > 1 else None,
            ensemble_image_sizes=ensemble_image_sizes if len(ensemble_models) > 1 else None,
        )
    else:
        primary = _run_tiled_eval(
            args,
            sam,
            multimask_output,
            test_save_path,
            ensemble_models=ensemble_models if len(ensemble_models) > 1 else None,
            ensemble_image_sizes=ensemble_image_sizes if len(ensemble_models) > 1 else None,
        )
        if args.legacy_box_eval:
            _run_legacy_eval(args, sam, multimask_output, None)

    logging.info(
        "Primary evaluation finished. best_threshold=%.2f save_threshold=%.2f mean_iou=%.4f",
        float(primary["best_threshold"]),
        float(primary["save_threshold"]),
        float(primary["best_metric"][3]),
    )
    logging.info("Testing Finished!")
