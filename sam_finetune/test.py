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
        resolve_predict_mode,
        resolve_predict_threshold,
        resolve_tile_settings,
    )
    from tiled_inference import (
        best_threshold_result,
        continuity_metrics,
        tiled_model_score_map,
        threshold_sweep,
    )
    from utils import test_single_volume
except ImportError:
    from .datasets.dataset_generic import GenericDataset, ValGenerator, list_image_files, load_image_mask_arrays
    from .runtime import (
        apply_delta_to_sam,
        resolve_predict_mode,
        resolve_predict_threshold,
        resolve_tile_settings,
    )
    from .tiled_inference import (
        best_threshold_result,
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


def _run_tiled_eval(args, model, multimask_output, test_save_path=None):
    case_names = list_image_files(os.path.join(args.volume_path, "images"))
    logging.info("%d tiled full-image test iterations", len(case_names))

    metric_by_thr = {float(thr): np.zeros(4, dtype=np.float64) for thr in args.val_thresholds}
    total_cases = 0
    saved_cases = []
    tile_size, tile_overlap = resolve_tile_settings(args.delta_ckpt, args.tile_size, args.tile_overlap)

    for i_batch, case_name in enumerate(case_names):
        image_hwc, label = load_image_mask_arrays(args.volume_path, case_name)
        score_map = tiled_model_score_map(
            image_hwc,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            model=model,
            image_size=int(args.img_size),
            multimask_output=multimask_output,
            use_amp=False,
        )
        sweep = threshold_sweep(score_map, label, args.val_thresholds)
        total_cases += 1
        case_logs = []
        for thr in args.val_thresholds:
            metric_by_thr[float(thr)] += np.array(sweep[float(thr)], dtype=np.float64)
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
        "skeleton_dice": 0.0,
        "centerline_precision": 0.0,
        "centerline_recall": 0.0,
        "component_fragmentation": 0.0,
    }

    if test_save_path is not None:
        for case_name, image_hwc, prob_map, label in saved_cases:
            _save_case_outputs(test_save_path, case_name, image_hwc, prob_map, label, save_threshold)
            metrics = continuity_metrics((prob_map >= float(best_thr)).astype(np.uint8), label)
            for key in continuity_totals:
                continuity_totals[key] += float(metrics[key])
    else:
        for case_name in case_names:
            image_hwc, label = load_image_mask_arrays(args.volume_path, case_name)
            score_map = tiled_model_score_map(
                image_hwc,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                model=model,
                image_size=int(args.img_size),
                multimask_output=multimask_output,
                use_amp=False,
            )
            metrics = continuity_metrics((score_map >= float(best_thr)).astype(np.uint8), label)
            for key in continuity_totals:
                continuity_totals[key] += float(metrics[key])

    continuity_totals = {key: value / max(1, total_cases) for key, value in continuity_totals.items()}

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
    parser.add_argument("--pred_threshold", default="auto", help="Probability threshold as float or 'auto'")
    parser.add_argument("--val_thresholds", type=float, nargs="+", default=[0.35, 0.4, 0.45, 0.5, 0.55, 0.6], help="Candidate probability thresholds")
    parser.add_argument("--eval_mode", default="auto", choices=["auto", "tile_full_box", "legacy_full_box"], help="Primary evaluation mode")
    parser.add_argument("--tile_size", type=int, default=-1, help="Tile size for tiled evaluation (-1 = use checkpoint metadata or 512)")
    parser.add_argument("--tile_overlap", type=int, default=-1, help="Tile overlap for tiled evaluation (-1 = use checkpoint metadata or tile_size // 2)")
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

    sam, _ = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=NUM_CLASSES,
        checkpoint=args.ckpt,
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
    )
    apply_delta_to_sam(
        sam=sam,
        delta_type=args.delta_type,
        delta_ckpt_path=args.delta_ckpt,
        middle_dim=int(args.middle_dim),
        scaling_factor=float(args.scaling_factor),
        rank=int(args.rank),
    )
    sam = sam.cuda()
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
    else:
        primary = _run_tiled_eval(args, sam, multimask_output, test_save_path)
        if args.legacy_box_eval:
            _run_legacy_eval(args, sam, multimask_output, None)

    logging.info(
        "Primary evaluation finished. best_threshold=%.2f save_threshold=%.2f mean_iou=%.4f",
        float(primary["best_threshold"]),
        float(primary["save_threshold"]),
        float(primary["best_metric"][3]),
    )
    logging.info("Testing Finished!")
