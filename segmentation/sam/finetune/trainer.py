import json
import logging
import os
import random
import sys

import numpy as np
from torch_runtime import DataLoader, get_torch_utils_data
from torch_runtime import nn, optim, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from ...shared import SQLiteLogHandler, SQLiteRunStore

try:
    from .tiled_inference import (
        best_threshold_result,
        continuity_metrics,
        resolve_tile_overlap,
        threshold_sweep,
        tiled_model_score_map,
    )
    from .utils import (
        BinaryCenterlineDiceLoss,
        BinaryClDiceLoss,
        BinaryDiceLoss,
        BinaryFocalWithLogitsLoss,
        BinaryTverskyLoss,
        centerline_target_batch,
        test_single_volume,
    )
except ImportError:
    from .tiled_inference import (
        best_threshold_result,
        continuity_metrics,
        resolve_tile_overlap,
        threshold_sweep,
        tiled_model_score_map,
    )
    from .utils import (
        BinaryCenterlineDiceLoss,
        BinaryClDiceLoss,
        BinaryDiceLoss,
        BinaryFocalWithLogitsLoss,
        BinaryTverskyLoss,
        centerline_target_batch,
        test_single_volume,
    )


PROMPT_SCHEDULES = {
    "hybrid_v1": (0.50, 0.30, 0.20),
    "hybrid_val_aligned": (0.20, 0.55, 0.25),
    "hybrid_val_balanced": (0.30, 0.40, 0.30),
    "hybrid_tight_heavy": (0.35, 0.25, 0.40),
    "points_heavy": (0.60, 0.10, 0.30),
}

_MASK_INDEX_CACHE: dict[str, dict[str, str]] = {}
_RUN_STORE: SQLiteRunStore | None = None


def _dataset_api():
    from ...datasets import sam_finetune as sam_datasets

    return sam_datasets


def _dist_is_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _dist_is_ready() else 0


def _get_world_size() -> int:
    return dist.get_world_size() if _dist_is_ready() else 1


def _is_main_process() -> bool:
    return _get_rank() == 0


def _barrier() -> None:
    if _dist_is_ready():
        dist.barrier()


def _reduce_mean_scalar(value: float, *, device: torch.device) -> float:
    if not _dist_is_ready():
        return float(value)
    tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(_get_world_size())
    return float(tensor.item())


def _reduce_sum_scalar(value: float, *, device: torch.device) -> float:
    if not _dist_is_ready():
        return float(value)
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _init_csv(path: str, headers: list[str]) -> None:
    if not path:
        return
    if _RUN_STORE is not None:
        _RUN_STORE.ensure_table(path, headers)
        return
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def _append_csv_rows(path: str, rows: list[list]) -> None:
    if not rows:
        return
    if _RUN_STORE is not None:
        _RUN_STORE.insert_rows(path, rows)
        return
    import csv

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def _append_csv_row(path: str, row: list) -> None:
    if _RUN_STORE is not None:
        _RUN_STORE.insert_row(path, row)
        return
    import csv

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def calc_loss(
    outputs,
    high_res_label_batch,
    bce_loss,
    dice_loss,
    tversky_loss,
    focal_loss,
    cldice_loss,
    centerline_loss,
    *,
    bce_weight: float,
    dice_weight: float,
    tversky_weight: float,
    focal_weight: float,
    cldice_weight: float,
    centerline_aux_weight: float,
):
    masks = outputs["masks"]
    if masks.shape[1] == 1:
        logits = masks
    else:
        scores = outputs.get("iou_predictions")
        if scores is None:
            logits = masks[:, :1]
        else:
            best_idx = torch.argmax(scores, dim=1)
            logits = masks[torch.arange(masks.shape[0], device=masks.device), best_idx].unsqueeze(1)

    targets = high_res_label_batch.float().unsqueeze(1)
    loss_bce = bce_loss(logits, targets)
    loss_dice = dice_loss(logits, high_res_label_batch)
    loss_tversky = tversky_loss(logits, high_res_label_batch)
    loss_focal = focal_loss(logits, high_res_label_batch)
    loss_cldice = cldice_loss(logits, high_res_label_batch) if float(cldice_weight) > 0.0 else logits.new_tensor(0.0)
    centerline_logits = outputs.get("centerline_logits", logits)
    if float(centerline_aux_weight) > 0.0:
        centerline_target = centerline_target_batch(high_res_label_batch)
        loss_centerline = centerline_loss(centerline_logits, centerline_target)
    else:
        loss_centerline = centerline_logits.new_tensor(0.0)
    loss = (
        float(bce_weight) * loss_bce
        + float(dice_weight) * loss_dice
        + float(tversky_weight) * loss_tversky
        + float(focal_weight) * loss_focal
        + float(cldice_weight) * loss_cldice
        + float(centerline_aux_weight) * loss_centerline
    )
    return loss, loss_bce, loss_dice, loss_tversky, loss_focal, loss_cldice, loss_centerline


def _safe_thresholds(args) -> list[float]:
    values = getattr(args, "val_thresholds", None) or [0.5]
    out = []
    for v in values:
        v = float(v)
        if 0.0 < v < 1.0:
            out.append(v)
    return sorted(set(out)) or [0.5]


def worker_init_fn(worker_id):
    seed = int(torch.initial_seed() % (2 ** 32))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _find_mask_path(mask_dir: str, base_name: str) -> str | None:
    mask_dir = os.path.abspath(mask_dir)
    mask_index = _MASK_INDEX_CACHE.get(mask_dir)
    if mask_index is None:
        mask_index = _dataset_api().get_mask_index(mask_dir)
        _MASK_INDEX_CACHE[mask_dir] = mask_index
    return mask_index.get(str(base_name))


def _estimate_pos_weight(mask_dir, sample_list, sample_size=200, min_weight=1.0, max_weight=20.0):
    names = list(sample_list or [])
    if not names:
        return None
    if sample_size and len(names) > int(sample_size):
        names = random.sample(names, int(sample_size))

    pos = 0
    total = 0
    used = 0
    import cv2

    for img_name in names:
        base_name = os.path.splitext(img_name)[0]
        mask_path = _find_mask_path(mask_dir, base_name)
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


def _estimate_pos_weight_across_roots(roots, sample_size=200, min_weight=1.0, max_weight=20.0):
    list_image_files = _dataset_api().list_image_files
    weighted_ratio = 0.0
    used_total = 0
    counted_roots = 0
    for root in roots:
        mask_dir = os.path.join(root, "masks")
        img_dir = os.path.join(root, "images")
        if not os.path.isdir(mask_dir) or not os.path.isdir(img_dir):
            continue
        sample_list = list_image_files(img_dir)
        result = _estimate_pos_weight(
            mask_dir=mask_dir,
            sample_list=sample_list,
            sample_size=sample_size,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        if result is None:
            continue
        _weight, pos_ratio, used = result
        if used <= 0:
            continue
        weighted_ratio += float(pos_ratio) * float(used)
        used_total += int(used)
        counted_roots += 1

    if counted_roots == 0 or used_total == 0:
        return None
    pos_ratio = weighted_ratio / float(used_total)
    if pos_ratio <= 0:
        weight = float(max_weight)
    else:
        weight = (1.0 - pos_ratio) / pos_ratio
    weight = max(float(min_weight), min(float(max_weight), float(weight)))
    return float(weight), float(pos_ratio), int(used_total)


def _save_delta_model(model, path: str) -> None:
    try:
        model.save_delta_parameters(path)
    except Exception:
        model.module.save_delta_parameters(path)


def _write_inference_config(snapshot_path: str, args, best_threshold: float | None) -> None:
    tile_overlap = resolve_tile_overlap(int(args.img_size), getattr(args, "tile_overlap", -1))
    prompt_policy = str(getattr(args, "prompt_policy", "hybrid_v1")).lower().strip()
    if prompt_policy == "hybrid":
        prompt_policy = "hybrid_v1"
    payload = {
        "img_size": int(args.img_size),
        "tile_overlap": int(tile_overlap),
        "best_threshold": None if best_threshold is None else float(best_threshold),
        "entry_module": "segmentation.sam.finetune.train",
        "decoder_type": str(getattr(args, "decoder_type", "baseline")).strip().lower(),
        "primary_val_mode": "tile_full_box",
        "prompt_schedule": "legacy_v1" if prompt_policy == "legacy" else prompt_policy,
        "predict_mode": "tile_full_box",
        "merge_mode": "score_weighted_center_blend",
        "profile": str(getattr(args, "profile", "custom")).strip().lower(),
        "augment_profile": str(getattr(args, "augment_profile", "balanced")).strip().lower(),
        "crop_policy": str(getattr(args, "crop_policy", "smart")).strip().lower(),
        "world_size": int(getattr(args, "world_size", 1)),
        "per_gpu_batch_size": int(getattr(args, "batch_size", 1)),
        "global_batch_size": int(getattr(args, "global_batch_size", getattr(args, "batch_size", 1))),
        "tile_batch_size": int(getattr(args, "tile_batch_size", 1)),
        "pipeline_stage": str(getattr(args, "pipeline_stage", "coarse")).strip().lower(),
        "refine_enabled": bool(getattr(args, "refine_enabled", False)),
        "refine_tile_size": int(getattr(args, "refine_tile_size", getattr(args, "roi_size", 768))),
        "refine_tile_sizes": [int(v) for v in (getattr(args, "refine_tile_sizes", None) or []) if int(v) > 0],
        "refine_max_rois": int(getattr(args, "refine_max_rois", 16)),
        "refine_roi_padding": int(getattr(args, "refine_roi_padding", 64)),
        "refine_merge_mode": str(getattr(args, "refine_merge_mode", "weighted_replace")).strip().lower(),
        "refine_score_threshold": float(getattr(args, "refine_score_threshold", 0.15)),
        "refine_batch_size": int(getattr(args, "refine_batch_size", getattr(args, "tile_batch_size", 1))),
        "refine_positive_band_low": float(getattr(args, "roi_positive_band_low", 0.20)),
        "refine_positive_band_high": float(getattr(args, "roi_positive_band_high", 0.90)),
        "centerline_head": bool(getattr(args, "centerline_head", False)),
    }
    with open(os.path.join(snapshot_path, "inference_config.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _prepare_legacy_prompts(sampled_batch, *, use_boxes: bool, use_points: bool, device):
    box_batch = sampled_batch["box"].to(device=device) if use_boxes and "box" in sampled_batch else None
    if use_points and "point_coords" in sampled_batch and "point_labels" in sampled_batch:
        point_coords_batch = sampled_batch["point_coords"].to(device=device)
        point_labels_batch = sampled_batch["point_labels"].to(device=device)
        return box_batch, (point_coords_batch, point_labels_batch)
    return box_batch, None


def _normalize_prompt_policy(prompt_policy: str) -> str:
    normalized = str(prompt_policy or "hybrid_v1").strip().lower()
    if normalized == "hybrid":
        return "hybrid_v1"
    return normalized


def _prepare_hybrid_prompts(sampled_batch, *, device, prompt_policy: str):
    full_box = sampled_batch["full_box"].to(device=device)
    tight_box = sampled_batch["tight_box"].to(device=device)
    pos_point = sampled_batch["pos_point"].to(device=device)
    neg_point = sampled_batch["neg_point"].to(device=device)
    has_foreground = sampled_batch["has_foreground"].to(device=device).bool()
    schedule_key = _normalize_prompt_policy(prompt_policy)
    full_points_prob, full_only_prob, _tight_points_prob = PROMPT_SCHEDULES.get(schedule_key, PROMPT_SCHEDULES["hybrid_v1"])

    box_batch = full_box.clone()
    point_coords = torch.zeros((full_box.shape[0], 2, 2), device=device, dtype=torch.float32)
    point_labels = -torch.ones((full_box.shape[0], 2), device=device, dtype=torch.float32)

    prompt_counts = {
        "full_box_points": 0,
        "full_box_only": 0,
        "tight_box_points": 0,
        "background_full_box_only": 0,
    }

    for index in range(full_box.shape[0]):
        if not bool(has_foreground[index].item()):
            box_batch[index] = full_box[index]
            prompt_counts["background_full_box_only"] += 1
            continue

        choice = random.random()
        if choice < full_points_prob:
            box_batch[index] = full_box[index]
            point_coords[index, 0] = pos_point[index]
            point_coords[index, 1] = neg_point[index]
            point_labels[index] = torch.tensor([1.0, 0.0], device=device)
            prompt_counts["full_box_points"] += 1
        elif choice < full_points_prob + full_only_prob:
            box_batch[index] = full_box[index]
            prompt_counts["full_box_only"] += 1
        else:
            box_batch[index] = tight_box[index]
            point_coords[index, 0] = pos_point[index]
            point_coords[index, 1] = neg_point[index]
            point_labels[index] = torch.tensor([1.0, 0.0], device=device)
            prompt_counts["tight_box_points"] += 1

    points_batch = None
    if torch.any(point_labels >= 0):
        points_batch = (point_coords, point_labels)
    return box_batch, points_batch, prompt_counts


def _select_logits(outputs):
    masks = outputs["masks"]
    if masks.shape[1] == 1:
        return masks[:, :1]
    scores = outputs.get("iou_predictions")
    if scores is None:
        return masks[:, :1]
    best_idx = torch.argmax(scores, dim=1)
    return masks[torch.arange(masks.shape[0], device=masks.device), best_idx].unsqueeze(1)


def _run_tiled_full_image_eval(
    val_root,
    case_names,
    model,
    *,
    tile_size: int,
    tile_overlap: int,
    thresholds: list[float],
    image_size: int,
    multimask_output: bool,
    use_amp: bool,
    tile_batch_size: int = 1,
    compute_continuity: bool = True,
    case_metrics_csv_path: str | None = None,
    epoch_num: int | None = None,
):
    load_image_mask_arrays = _dataset_api().load_image_mask_arrays
    metric_by_thr = {float(thr): np.zeros(4, dtype=np.float64) for thr in thresholds}
    continuity_by_thr = (
        {
            float(thr): {
                "skeleton_dice": 0.0,
                "centerline_precision": 0.0,
                "centerline_recall": 0.0,
                "component_fragmentation": 0.0,
            }
            for thr in thresholds
        }
        if compute_continuity
        else None
    )
    total_cases = 0

    for i_batch, case_name in enumerate(tqdm(case_names)):
        image_hwc, label = load_image_mask_arrays(val_root, case_name)
        score_map = tiled_model_score_map(
            image_hwc,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            model=model,
            image_size=image_size,
            multimask_output=multimask_output,
            use_amp=use_amp,
            tile_batch_size=tile_batch_size,
        )
        sweep = threshold_sweep(score_map, label, thresholds)
        total_cases += 1
        case_rows = []
        for thr in thresholds:
            metric_by_thr[float(thr)] += np.array(sweep[float(thr)], dtype=np.float64)
            if continuity_by_thr is not None:
                continuity = continuity_metrics((score_map >= float(thr)).astype(np.uint8), label)
                for key in continuity_by_thr[float(thr)]:
                    continuity_by_thr[float(thr)][key] += float(continuity[key])
            if case_metrics_csv_path is not None:
                metrics = sweep[float(thr)]
                case_rows.append(
                    [
                        int(epoch_num) if epoch_num is not None else -1,
                        int(i_batch),
                        str(case_name),
                        float(thr),
                        float(metrics[0]),
                        float(metrics[1]),
                        float(metrics[2]),
                        float(metrics[3]),
                    ]
                )
        _append_csv_rows(case_metrics_csv_path, case_rows)

    metric_by_thr = {
        float(thr): metric_by_thr[float(thr)] / max(1, total_cases)
        for thr in thresholds
    }
    if continuity_by_thr is not None:
        continuity_by_thr = {
            float(thr): {
                key: float(value) / max(1, total_cases)
                for key, value in continuity_by_thr[float(thr)].items()
            }
            for thr in thresholds
        }
    selected_thr, metric = best_threshold_result(metric_by_thr)
    return selected_thr, metric, metric_by_thr, continuity_by_thr, total_cases


def _run_tiled_continuity_eval(
    val_root,
    case_names,
    model,
    *,
    tile_size: int,
    tile_overlap: int,
    threshold: float,
    image_size: int,
    multimask_output: bool,
    use_amp: bool,
):
    load_image_mask_arrays = _dataset_api().load_image_mask_arrays
    totals = {
        "skeleton_dice": 0.0,
        "centerline_precision": 0.0,
        "centerline_recall": 0.0,
        "component_fragmentation": 0.0,
    }
    total_cases = 0
    for case_name in tqdm(case_names):
        image_hwc, label = load_image_mask_arrays(val_root, case_name)
        score_map = tiled_model_score_map(
            image_hwc,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            model=model,
            image_size=image_size,
            multimask_output=multimask_output,
            use_amp=use_amp,
        )
        pred_mask = (score_map >= float(threshold)).astype(np.uint8)
        metrics = continuity_metrics(pred_mask, label)
        for key in totals:
            totals[key] += float(metrics[key])
        total_cases += 1
    if total_cases == 0:
        return {key: 0.0 for key in totals}
    return {key: float(value) / float(total_cases) for key, value in totals.items()}


def _run_legacy_box_eval(
    valloader,
    model,
    args,
    multimask_output,
    thresholds,
    *,
    case_metrics_csv_path: str | None = None,
    epoch_num: int | None = None,
):
    metric_by_thr = {float(thr): np.zeros(4, dtype=np.float64) for thr in thresholds}
    total_cases = 0

    for i_batch, sampled_batch in enumerate(tqdm(valloader)):
        image = sampled_batch["image"]
        label = sampled_batch["label"]
        case_name = sampled_batch["case_name"][0]
        box_only = None
        if not bool(getattr(args, "full_image_eval", False)):
            box_only, _ = _prepare_legacy_prompts(
                sampled_batch,
                use_boxes=True,
                use_points=False,
                device=torch.device("cuda"),
            )
        case_rows = []
        for thr in thresholds:
            metric_box_i = test_single_volume(
                image,
                label,
                model,
                classes=1,
                multimask_output=multimask_output,
                patch_size=[args.img_size, args.img_size],
                test_save_path=None,
                boxes=box_only,
                points=None,
                use_full_image_box_prompt=bool(getattr(args, "full_image_eval", False)),
                threshold_prob=float(thr),
            )
            metric_by_thr[float(thr)] += np.array(metric_box_i, dtype=np.float64)
            if case_metrics_csv_path is not None:
                case_rows.append(
                    [
                        int(epoch_num) if epoch_num is not None else -1,
                        int(i_batch),
                        str(case_name),
                        float(thr),
                        float(metric_box_i[0]),
                        float(metric_box_i[1]),
                        float(metric_box_i[2]),
                        float(metric_box_i[3]),
                    ]
                )
        total_cases += 1
        _append_csv_rows(case_metrics_csv_path, case_rows)

    metric_by_thr = {
        float(thr): metric_by_thr[float(thr)] / max(1, total_cases)
        for thr in thresholds
    }
    selected_thr, metric = best_threshold_result(metric_by_thr)
    return selected_thr, metric, metric_by_thr, total_cases


def _build_train_dataset(args, *, split: str, transform, patches_per_image: int, use_full_image_box: bool):
    GenericDataset = _dataset_api().GenericDataset
    utils_data = get_torch_utils_data()
    roots = [args.root_path]
    for collection in (
        getattr(args, "extra_train_roots", None) or [],
        getattr(args, "pseudo_label_roots", None) or [],
    ):
        for root in collection:
            if root and str(root).strip():
                roots.append(str(root).strip())

    datasets = [
        GenericDataset(
            base_dir=root,
            split=split,
            transform=transform,
            patches_per_image=patches_per_image,
            use_full_image_box=use_full_image_box,
        )
        for root in roots
    ]
    if len(datasets) == 1:
        return datasets[0], roots
    return utils_data.ConcatDataset(datasets), roots


def trainer_generic(args, model, snapshot_path, multimask_output, low_res):
    global _RUN_STORE
    datasets_api = _dataset_api()
    GenericDataset = datasets_api.GenericDataset
    RandomGenerator = datasets_api.RandomGenerator
    RefineRandomGenerator = datasets_api.RefineRandomGenerator
    ValGenerator = datasets_api.ValGenerator
    list_image_files = datasets_api.list_image_files
    device = torch.device("cuda", int(getattr(args, "local_rank", 0)) if bool(getattr(args, "distributed", False)) else 0)
    is_main_process = _is_main_process()
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    if is_main_process:
        _RUN_STORE = SQLiteRunStore(os.path.join(snapshot_path, "training.sqlite3"))
        sqlite_handler = SQLiteLogHandler(_RUN_STORE, table_name="logs", flush_every=20)
        sqlite_handler.setFormatter(formatter)
        logger.addHandler(sqlite_handler)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    logging.info(str(args))

    base_lr = args.base_lr
    batch_size = int(args.batch_size)
    tile_overlap = resolve_tile_overlap(int(args.img_size), getattr(args, "tile_overlap", -1))
    prompt_policy = _normalize_prompt_policy(getattr(args, "prompt_policy", "hybrid_v1"))
    pipeline_stage = str(getattr(args, "pipeline_stage", "coarse")).strip().lower()
    val_case_names = list_image_files(os.path.join(args.val_path, "images"))
    train_patch_size = int(getattr(args, "roi_size", args.img_size)) if pipeline_stage == "refine" else int(args.img_size)
    refine_patches_per_image = 4 if pipeline_stage == "refine" else int(args.patches_per_image)
    train_generator_cls = RefineRandomGenerator if pipeline_stage == "refine" else RandomGenerator

    db_val_legacy = GenericDataset(
        base_dir=args.val_path,
        split="val_vol",
        transform=ValGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res]),
    )
    train_transform = train_generator_cls(
        output_size=[train_patch_size, train_patch_size],
        low_res=[low_res, low_res],
        background_crop_prob=args.background_crop_prob,
        near_background_crop_prob=args.near_background_crop_prob,
        hard_negative_crop_prob=getattr(args, "hard_negative_crop_prob", 0.10),
        augment_profile=getattr(args, "augment_profile", "balanced"),
        crop_policy=getattr(args, "crop_policy", "smart"),
        **(
            {
                "roi_positive_band_low": float(getattr(args, "roi_positive_band_low", 0.20)),
                "roi_positive_band_high": float(getattr(args, "roi_positive_band_high", 0.90)),
            }
            if pipeline_stage == "refine"
            else {}
        ),
    )
    db_train, train_roots = _build_train_dataset(
        args,
        split="train",
        transform=train_transform,
        patches_per_image=refine_patches_per_image,
        use_full_image_box=bool(getattr(args, "train_full_image_box", False)),
    )

    if is_main_process:
        print("The length of train set is: {}".format(len(db_train)))

    train_sampler = None
    if bool(getattr(args, "distributed", False)):
        train_sampler = DistributedSampler(
            db_train,
            num_replicas=int(getattr(args, "world_size", _get_world_size())),
            rank=int(getattr(args, "global_rank", _get_rank())),
            shuffle=True,
            drop_last=False,
        )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": train_sampler is None,
        "sampler": train_sampler,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "worker_init_fn": worker_init_fn,
    }
    val_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": max(1, args.num_workers // 2) if args.num_workers > 0 else 0,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 32
        if val_kwargs["num_workers"] > 0:
            val_kwargs["persistent_workers"] = True
            val_kwargs["prefetch_factor"] = 16

    trainloader = DataLoader(db_train, **loader_kwargs)
    valloader_legacy = DataLoader(db_val_legacy, **val_kwargs)

    if bool(getattr(args, "distributed", False)):
        model = DDP(
            model,
            device_ids=[int(getattr(args, "local_rank", 0))],
            output_device=int(getattr(args, "local_rank", 0)),
            find_unused_parameters=False,
        )
    elif args.n_gpu > 1:
        model = nn.DataParallel(model)

    pos_weight_value = args.pos_weight
    auto_requested = isinstance(pos_weight_value, str) and str(pos_weight_value).lower() == "auto"
    if auto_requested:
        auto_info = _estimate_pos_weight_across_roots(
            train_roots,
            sample_size=int(getattr(args, "pos_weight_sample", 200)),
            min_weight=float(getattr(args, "pos_weight_min", 1.0)),
            max_weight=float(getattr(args, "pos_weight_max", 20.0)),
        )
        if auto_info is None:
            pos_weight_value = 1.0
            logging.info("Auto pos_weight failed. Falling back to 1.0.")
        else:
            pos_weight_value, pos_ratio, used = auto_info
            logging.info(
                "Auto pos_weight: %.2f (pos_ratio=%.4f, samples=%d)"
                % (float(pos_weight_value), float(pos_ratio), int(used))
            )
    else:
        pos_weight_value = float(pos_weight_value)

    pos_weight_tensor = torch.tensor([float(pos_weight_value)], device=device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    dice_loss = BinaryDiceLoss()
    tversky_loss = BinaryTverskyLoss(alpha=args.tversky_alpha, beta=args.tversky_beta)
    focal_loss = BinaryFocalWithLogitsLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    cldice_loss = BinaryClDiceLoss(iterations=int(getattr(args, "cldice_iters", 20)))
    centerline_loss = BinaryCenterlineDiceLoss()
    val_thresholds = _safe_thresholds(args)
    b_lr = base_lr / args.warmup_period if args.warmup else base_lr
    decoder_lr_mult = float(getattr(args, "decoder_lr_mult", 0.1))

    lora_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "mask_decoder" in name:
            decoder_params.append(param)
        else:
            lora_params.append(param)

    if args.AdamW:
        optimizer = optim.AdamW(
            [
                {"params": lora_params, "lr": b_lr},
                {"params": decoder_params, "lr": b_lr * decoder_lr_mult},
            ],
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
    else:
        optimizer = optim.SGD(
            [
                {"params": lora_params, "lr": b_lr},
                {"params": decoder_params, "lr": b_lr * decoder_lr_mult},
            ],
            momentum=0.9,
            weight_decay=0.0001,
        )

    if is_main_process:
        print(f"Detected {len(lora_params)} param tensors for LoRA (LR: {b_lr})")
        print(f"Detected {len(decoder_params)} param tensors for Decoder (LR: {b_lr * decoder_lr_mult})")

    from torch.optim.lr_scheduler import CosineAnnealingLR

    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp) if args.use_amp else None
    iter_num = 0

    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("%d iterations per epoch. %d max iterations " % (len(trainloader), max_iterations))
    csv_path = "val_summary"
    train_step_csv_path = "train_steps"
    train_epoch_csv_path = "train_epochs"
    val_threshold_csv_path = "val_thresholds"
    val_tile_case_csv_path = "val_tile_cases"
    val_legacy_case_csv_path = "val_legacy_cases"
    if is_main_process:
        _init_csv(
            csv_path,
            [
                "epoch",
                "tile_threshold",
                "tile_precision",
                "tile_recall",
                "tile_dice",
                "tile_iou",
                "tile_skeleton_dice",
                "tile_centerline_precision",
                "tile_centerline_recall",
                "tile_component_fragmentation",
                "legacy_threshold",
                "legacy_precision",
                "legacy_recall",
                "legacy_dice",
                "legacy_iou",
            ],
        )
        _init_csv(
            train_step_csv_path,
            ["epoch", "iteration", "loss", "loss_bce", "loss_dice", "loss_tversky", "loss_focal", "loss_cldice", "loss_centerline", "lr"],
        )
        _init_csv(
            train_epoch_csv_path,
            [
                "epoch",
                "mean_loss",
                "mean_loss_bce",
                "mean_loss_dice",
                "mean_loss_tversky",
                "mean_loss_focal",
                "mean_loss_cldice",
                "mean_loss_centerline",
                "lr",
                "full_box_points",
                "full_box_only",
                "tight_box_points",
                "background_full_box_only",
            ],
        )
        _init_csv(
            val_threshold_csv_path,
            ["epoch", "mode", "threshold", "precision", "recall", "dice", "iou"],
        )
        _init_csv(
            val_tile_case_csv_path,
            ["epoch", "case_index", "case_name", "threshold", "precision", "recall", "dice", "iou"],
        )
        _init_csv(
            val_legacy_case_csv_path,
            ["epoch", "case_index", "case_name", "threshold", "precision", "recall", "dice", "iou"],
        )
        logging.info(
            "Effective training profile: pipeline_stage=%s prompt_policy=%s augment_profile=%s crop_policy=%s background_crop_prob=%.3f near_background_crop_prob=%.3f hard_negative_crop_prob=%.3f decoder_lr_mult=%.3f hq_trainable_mode=%s tversky_alpha=%.3f tversky_beta=%.3f cldice_weight=%.3f centerline_aux_weight=%.3f train_roots=%s"
            % (
                pipeline_stage,
                prompt_policy,
                str(getattr(args, "augment_profile", "balanced")),
                str(getattr(args, "crop_policy", "smart")),
                float(getattr(args, "background_crop_prob", 0.2)),
                float(getattr(args, "near_background_crop_prob", 0.15)),
                float(getattr(args, "hard_negative_crop_prob", 0.10)),
                float(decoder_lr_mult),
                str(getattr(args, "hq_trainable_mode", "balanced")),
                float(getattr(args, "tversky_alpha", 0.3)),
                float(getattr(args, "tversky_beta", 0.7)),
                float(getattr(args, "cldice_weight", 0.0)),
                float(getattr(args, "centerline_aux_weight", 0.0)),
                    ",".join(train_roots),
                )
            )

    iterator = tqdm(range(max_epoch), ncols=70, disable=not is_main_process)
    best_performance = 0.0
    best_threshold = None
    patience = max_epoch
    patience_counter = 0
    stop_training = False
    train_step_buffer: list[list[float | int]] = []
    train_step_flush_every = 32
    if is_main_process:
        _write_inference_config(snapshot_path, args, best_threshold)

    for epoch_num in iterator:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_num)
        model.train()
        epoch_prompt_counts = {
            "full_box_points": 0,
            "full_box_only": 0,
            "tight_box_points": 0,
            "background_full_box_only": 0,
        }
        epoch_loss_sum = 0.0
        epoch_loss_bce_sum = 0.0
        epoch_loss_dice_sum = 0.0
        epoch_loss_tversky_sum = 0.0
        epoch_loss_focal_sum = 0.0
        epoch_loss_cldice_sum = 0.0
        epoch_loss_centerline_sum = 0.0
        epoch_step_count = 0
        for sampled_batch in trainloader:
            optimizer.zero_grad()
            image_batch = sampled_batch["image"].cuda()
            label_batch = sampled_batch["label"].cuda()

            if prompt_policy == "legacy":
                use_points = random.random() < float(getattr(args, "train_use_points_prob", 0.0))
                box_batch, points_batch = _prepare_legacy_prompts(
                    sampled_batch,
                    use_boxes=bool(getattr(args, "train_use_boxes", True)),
                    use_points=use_points,
                    device=image_batch.device,
                )
            else:
                box_batch, points_batch, prompt_counts = _prepare_hybrid_prompts(
                    sampled_batch,
                    device=image_batch.device,
                    prompt_policy=prompt_policy,
                )
                for key, value in prompt_counts.items():
                    epoch_prompt_counts[key] += int(value)

            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.use_amp):
                    outputs = model(image_batch, multimask_output, args.img_size, boxes=box_batch, points=points_batch)
                    loss, loss_bce, loss_dice, loss_tversky, loss_focal, loss_cldice_value_t, loss_centerline_value_t = calc_loss(
                        outputs,
                        label_batch,
                        bce_loss,
                        dice_loss,
                        tversky_loss,
                        focal_loss,
                        cldice_loss,
                        centerline_loss,
                        bce_weight=args.bce_weight,
                        dice_weight=args.dice_weight,
                        tversky_weight=args.tversky_weight,
                        focal_weight=args.focal_weight,
                        cldice_weight=getattr(args, "cldice_weight", 0.0),
                        centerline_aux_weight=getattr(args, "centerline_aux_weight", 0.0),
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(image_batch, multimask_output, args.img_size, boxes=box_batch, points=points_batch)
                loss, loss_bce, loss_dice, loss_tversky, loss_focal, loss_cldice_value_t, loss_centerline_value_t = calc_loss(
                    outputs,
                    label_batch,
                    bce_loss,
                    dice_loss,
                    tversky_loss,
                    focal_loss,
                    cldice_loss,
                    centerline_loss,
                    bce_weight=args.bce_weight,
                    dice_weight=args.dice_weight,
                    tversky_weight=args.tversky_weight,
                    focal_weight=args.focal_weight,
                    cldice_weight=getattr(args, "cldice_weight", 0.0),
                    centerline_aux_weight=getattr(args, "centerline_aux_weight", 0.0),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if args.warmup and iter_num < args.warmup_period:
                warmup_factor = (iter_num + 1) / args.warmup_period
                optimizer.param_groups[0]["lr"] = base_lr * warmup_factor
                if len(optimizer.param_groups) > 1:
                    optimizer.param_groups[1]["lr"] = (base_lr * decoder_lr_mult) * warmup_factor
                lr_ = optimizer.param_groups[0]["lr"]
            else:
                lr_ = optimizer.param_groups[0]["lr"]

            iter_num += 1
            loss_value = float(loss.item())
            loss_bce_value = float(loss_bce.item())
            loss_dice_value = float(loss_dice.item())
            loss_tversky_value = float(loss_tversky.item())
            loss_focal_value = float(loss_focal.item())
            loss_cldice_value = float(loss_cldice_value_t.item())
            loss_centerline_value = float(loss_centerline_value_t.item())
            epoch_loss_sum += loss_value
            epoch_loss_bce_sum += loss_bce_value
            epoch_loss_dice_sum += loss_dice_value
            epoch_loss_tversky_sum += loss_tversky_value
            epoch_loss_focal_sum += loss_focal_value
            epoch_loss_cldice_sum += loss_cldice_value
            epoch_loss_centerline_sum += loss_centerline_value
            epoch_step_count += 1
            if is_main_process:
                train_step_buffer.append(
                    [
                        int(epoch_num),
                        int(iter_num),
                        float(loss_value),
                        float(loss_bce_value),
                        float(loss_dice_value),
                        float(loss_tversky_value),
                        float(loss_focal_value),
                        float(loss_cldice_value),
                        float(loss_centerline_value),
                        float(lr_),
                    ]
                )
                if len(train_step_buffer) >= train_step_flush_every:
                    _append_csv_rows(train_step_csv_path, train_step_buffer)
                    train_step_buffer.clear()

        if is_main_process and train_step_buffer:
            _append_csv_rows(train_step_csv_path, train_step_buffer)
            train_step_buffer.clear()

        total_steps = _reduce_sum_scalar(epoch_step_count, device=device)
        mean_loss = _reduce_sum_scalar(epoch_loss_sum, device=device) / max(1.0, total_steps)
        mean_loss_bce = _reduce_sum_scalar(epoch_loss_bce_sum, device=device) / max(1.0, total_steps)
        mean_loss_dice = _reduce_sum_scalar(epoch_loss_dice_sum, device=device) / max(1.0, total_steps)
        mean_loss_tversky = _reduce_sum_scalar(epoch_loss_tversky_sum, device=device) / max(1.0, total_steps)
        mean_loss_focal = _reduce_sum_scalar(epoch_loss_focal_sum, device=device) / max(1.0, total_steps)
        mean_loss_cldice = _reduce_sum_scalar(epoch_loss_cldice_sum, device=device) / max(1.0, total_steps)
        mean_loss_centerline = _reduce_sum_scalar(epoch_loss_centerline_sum, device=device) / max(1.0, total_steps)
        prompt_counts_global = {
            key: int(_reduce_sum_scalar(value, device=device))
            for key, value in epoch_prompt_counts.items()
        }
        if is_main_process:
            _append_csv_rows(
                train_epoch_csv_path,
                [[
                    int(epoch_num),
                    float(mean_loss),
                    float(mean_loss_bce),
                    float(mean_loss_dice),
                    float(mean_loss_tversky),
                    float(mean_loss_focal),
                    float(mean_loss_cldice),
                    float(mean_loss_centerline),
                    float(lr_),
                    int(prompt_counts_global["full_box_points"]),
                    int(prompt_counts_global["full_box_only"]),
                    int(prompt_counts_global["tight_box_points"]),
                    int(prompt_counts_global["background_full_box_only"]),
                ]],
            )
            logging.info(
                "Epoch %d/%d train: loss=%.4f bce=%.4f dice=%.4f tversky=%.4f focal=%.4f cldice=%.4f centerline=%.4f lr=%.6f"
                % (
                    epoch_num + 1,
                    max_epoch,
                    mean_loss,
                    mean_loss_bce,
                    mean_loss_dice,
                    mean_loss_tversky,
                    mean_loss_focal,
                    mean_loss_cldice,
                    mean_loss_centerline,
                    float(lr_),
                )
            )

        if prompt_policy != "legacy" and is_main_process:
            logging.info(
                "Epoch %d prompt mix: full_box+points=%d full_box_only=%d tight_box+points=%d background_only=%d"
                % (
                    epoch_num + 1,
                    prompt_counts_global["full_box_points"],
                    prompt_counts_global["full_box_only"],
                    prompt_counts_global["tight_box_points"],
                    prompt_counts_global["background_full_box_only"],
                )
            )

        val_interval = args.save_interval
        if epoch_num % val_interval == 0 and is_main_process:
            model.eval()
            continuity_eval_interval = int(getattr(args, "continuity_eval_interval", 1))
            run_continuity = continuity_eval_interval > 0 and (epoch_num % continuity_eval_interval == 0)
            logging.info("%d tiled full-image val iterations per epoch" % len(val_case_names))
            tile_selected_thr, tile_metric, tile_by_thr, tile_continuity_by_thr, total_cases = _run_tiled_full_image_eval(
                args.val_path,
                val_case_names,
                model,
                tile_size=int(args.img_size),
                tile_overlap=tile_overlap,
                thresholds=val_thresholds,
                image_size=int(args.img_size),
                multimask_output=multimask_output,
                use_amp=bool(getattr(args, "use_amp", False)),
                tile_batch_size=int(getattr(args, "tile_batch_size", 1)),
                compute_continuity=run_continuity,
                case_metrics_csv_path=val_tile_case_csv_path,
                epoch_num=epoch_num,
            )
            if tile_continuity_by_thr is not None:
                tile_continuity = tile_continuity_by_thr[float(tile_selected_thr)]
            else:
                tile_continuity = {
                    "skeleton_dice": float("nan"),
                    "centerline_precision": float("nan"),
                    "centerline_recall": float("nan"),
                    "component_fragmentation": float("nan"),
                }
            threshold_rows = []
            for thr in val_thresholds:
                threshold_rows.append(
                    [
                        int(epoch_num),
                        "tile",
                        float(thr),
                        float(tile_by_thr[float(thr)][0]),
                        float(tile_by_thr[float(thr)][1]),
                        float(tile_by_thr[float(thr)][2]),
                        float(tile_by_thr[float(thr)][3]),
                    ]
                )
            _append_csv_rows(val_threshold_csv_path, threshold_rows)
            logging.info(
                "Epoch %d val(tile): thr=%.2f pr=%.4f re=%.4f f1=%.4f iou=%.4f"
                % (epoch_num + 1, tile_selected_thr, tile_metric[0], tile_metric[1], tile_metric[2], tile_metric[3])
            )
            logging.info(
                "Epoch %d continuity: thr=%.2f skeleton_dice=%.4f centerline_precision=%.4f centerline_recall=%.4f fragmentation=%.4f"
                % (
                    epoch_num + 1,
                    tile_selected_thr,
                    tile_continuity["skeleton_dice"],
                    tile_continuity["centerline_precision"],
                    tile_continuity["centerline_recall"],
                    tile_continuity["component_fragmentation"],
                )
            )
            if not run_continuity:
                logging.info("Continuity metrics skipped this cycle (continuity_eval_interval=%d)", continuity_eval_interval)
            logging.info("Validation used reconstructed full-image metrics over %d image(s)." % total_cases)

            legacy_selected_thr = float("nan")
            legacy_metric = np.array([float("nan")] * 4)
            if bool(getattr(args, "legacy_box_eval", False)):
                logging.info("%d legacy val iterations per epoch" % len(valloader_legacy))
                legacy_selected_thr, legacy_metric, legacy_by_thr, _legacy_cases = _run_legacy_box_eval(
                    valloader_legacy,
                    model,
                    args,
                    multimask_output,
                    val_thresholds,
                    case_metrics_csv_path=val_legacy_case_csv_path,
                    epoch_num=epoch_num,
                )
                legacy_rows = []
                for thr in val_thresholds:
                    legacy_rows.append(
                        [
                            int(epoch_num),
                            "legacy",
                            float(thr),
                            float(legacy_by_thr[float(thr)][0]),
                            float(legacy_by_thr[float(thr)][1]),
                            float(legacy_by_thr[float(thr)][2]),
                            float(legacy_by_thr[float(thr)][3]),
                        ]
                    )
                _append_csv_rows(val_threshold_csv_path, legacy_rows)
                logging.info(
                    "Epoch %d val(legacy): thr=%.2f pr=%.4f re=%.4f f1=%.4f iou=%.4f"
                    % (epoch_num + 1, legacy_selected_thr, legacy_metric[0], legacy_metric[1], legacy_metric[2], legacy_metric[3])
                )

            _append_csv_row(
                csv_path,
                [
                    epoch_num,
                    tile_selected_thr,
                    tile_metric[0],
                    tile_metric[1],
                    tile_metric[2],
                    tile_metric[3],
                    tile_continuity["skeleton_dice"],
                    tile_continuity["centerline_precision"],
                    tile_continuity["centerline_recall"],
                    tile_continuity["component_fragmentation"],
                    legacy_selected_thr,
                    legacy_metric[0],
                    legacy_metric[1],
                    legacy_metric[2],
                    legacy_metric[3],
                ],
            )

            performance = float(tile_metric[3])
            if performance > best_performance:
                best_performance = performance
                best_threshold = float(tile_selected_thr)
                patience_counter = 0
                save_best_path = os.path.join(snapshot_path, "best_model.pth")
                _save_delta_model(model, save_best_path)
                with open(os.path.join(snapshot_path, "best_threshold.txt"), "w", encoding="utf-8") as f:
                    f.write(f"{best_threshold:.4f}\n")
                _write_inference_config(snapshot_path, args, best_threshold)
                logging.info(
                    "New best tile_full_box IoU: %f at threshold %.2f. Saved best model to %s"
                    % (best_performance, best_threshold, save_best_path)
                )
            else:
                patience_counter += 1
                logging.info("Best tile_full_box IoU did not improve. Patience: %d/%d" % (patience_counter, patience))

            if patience_counter >= patience:
                logging.info("Early stopping triggered")
                stop_training = True

        scheduler.step()

        save_interval = args.save_interval
        if epoch_num % save_interval == 0 and is_main_process:
            save_mode_path = os.path.join(snapshot_path, "epoch_" + str(epoch_num) + ".pth")
            _save_delta_model(model, save_mode_path)
            logging.info("save model to %s" % save_mode_path)
        if _dist_is_ready():
            stop_tensor = torch.tensor(1 if stop_training else 0, device=device, dtype=torch.int32)
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            stop_training = bool(stop_tensor.item() > 0)
        _barrier()
        if stop_training:
            break

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            if is_main_process:
                save_mode_path = os.path.join(snapshot_path, "epoch_" + str(epoch_num) + ".pth")
                _save_delta_model(model, save_mode_path)
                logging.info("save model to %s" % save_mode_path)
            iterator.close()
            break

    if is_main_process and _RUN_STORE is not None:
        for handler in list(logger.handlers):
            try:
                handler.flush()
            except Exception:
                pass
        _RUN_STORE.close()
        _RUN_STORE = None

    return "Training Finished!"
