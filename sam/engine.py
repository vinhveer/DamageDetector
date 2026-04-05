from __future__ import annotations

import base64
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from torch_runtime import describe_device_fallback, select_device_str


_SAM_AUTO_MAX_SIDE = 2048
_SAM_AUTO_MAX_PIXELS = 4_194_304
_SAM_AUTO_RETRY_MAX_SIDE = 1280
_SAM_AUTO_RETRY_MAX_PIXELS = 1_572_864
_SAM_AUTO_MPS_QUALITY_MAX_SIDE = 1536
_SAM_AUTO_MPS_QUALITY_MAX_PIXELS = 2_359_296


@dataclass(frozen=True)
class SamParams:
    sam_checkpoint: str
    sam_model_type: str = "auto"
    invert_mask: bool = False
    sam_min_component_area: int = 0
    sam_dilate_iters: int = 0
    seed: int = 1337
    overlay_alpha: float = 0.45
    device: str = "auto"
    output_dir: str = "results_sam"
    roi_box: tuple[int, int, int, int] | None = None
    task_group: str = "crack_only"
    more_damage_max_masks: int = 8
    sam_auto_profile: str = "auto"
    sam_points_per_side: int = -1
    sam_points_per_batch: int = -1
    sam_pred_iou_thresh: float = -1.0
    sam_stability_score_thresh: float = -1.0
    sam_stability_score_offset: float = -1.0
    sam_box_nms_thresh: float = -1.0
    sam_crop_n_layers: int = -1
    sam_crop_overlap_ratio: float = -1.0
    sam_crop_nms_thresh: float = -1.0
    sam_crop_n_points_downscale_factor: int = -1
    sam_min_mask_region_area: int = -1
    selection_mode: str = "default"
    selection_prompt: str = ""


class SamRunner:
    def __init__(self) -> None:
        self._device: str | None = None
        self._sam_checkpoint: str | None = None
        self._sam_model_type: str | None = None
        self._predictor: Any | None = None

    def _ensure_import_paths(self) -> None:
        here = Path(__file__).resolve()
        repo_root = here.parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

    def ensure_model_loaded(self, params: SamParams, *, log_fn=None) -> tuple[Any, str]:
        self._ensure_import_paths()

        from sam.runtime import load_sam_model
        from segment_anything import SamPredictor

        device = select_device_str(params.device)
        fallback = describe_device_fallback(params.device, device)
        if fallback is not None and log_fn is not None:
            log_fn(fallback)
        requested_model_type = str(params.sam_model_type or "auto").strip().lower()
        needs_reload = (
            self._predictor is None
            or self._sam_checkpoint != params.sam_checkpoint
            or (requested_model_type != "auto" and self._sam_model_type != requested_model_type)
            or self._device != device
        )
        if not needs_reload:
            return self._predictor, device
        if log_fn is not None:
            log_fn("Loading SAM checkpoint...")
        sam_model, used_model_type = load_sam_model(params.sam_checkpoint, params.sam_model_type)
        sam_model.to(device=device)
        self._predictor = SamPredictor(sam_model)
        self._sam_checkpoint = params.sam_checkpoint
        self._sam_model_type = used_model_type
        self._device = device
        if log_fn is not None:
            log_fn(f"SAM ready (type={used_model_type}, device={device}).")
        return self._predictor, device

    def _run_with_roi(self, func_name: str, image_path: str, params: SamParams, **kwargs) -> dict:
        import cv2

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        roi_x1, roi_y1, roi_x2, roi_y2 = params.roi_box
        roi_x1 = max(0, min(image.shape[1] - 1, int(roi_x1)))
        roi_y1 = max(0, min(image.shape[0] - 1, int(roi_y1)))
        roi_x2 = max(0, min(image.shape[1], int(roi_x2)))
        roi_y2 = max(0, min(image.shape[0], int(roi_y2)))
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            raise ValueError("Invalid ROI box")
        crop = image[roi_y1:roi_y2, roi_x1:roi_x2]
        ext = os.path.splitext(image_path)[1] or ".png"
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        cv2.imwrite(tmp_path, crop)
        try:
            sub_params = SamParams(
                sam_checkpoint=params.sam_checkpoint,
                sam_model_type=params.sam_model_type,
                invert_mask=params.invert_mask,
                sam_min_component_area=params.sam_min_component_area,
                sam_dilate_iters=params.sam_dilate_iters,
                seed=params.seed,
                overlay_alpha=params.overlay_alpha,
                device=params.device,
                output_dir=params.output_dir,
                roi_box=None,
                task_group=params.task_group,
                more_damage_max_masks=params.more_damage_max_masks,
                sam_auto_profile=params.sam_auto_profile,
                sam_points_per_side=params.sam_points_per_side,
                sam_points_per_batch=params.sam_points_per_batch,
                sam_pred_iou_thresh=params.sam_pred_iou_thresh,
                sam_stability_score_thresh=params.sam_stability_score_thresh,
                sam_stability_score_offset=params.sam_stability_score_offset,
                sam_box_nms_thresh=params.sam_box_nms_thresh,
                sam_crop_n_layers=params.sam_crop_n_layers,
                sam_crop_overlap_ratio=params.sam_crop_overlap_ratio,
                sam_crop_nms_thresh=params.sam_crop_nms_thresh,
                sam_crop_n_points_downscale_factor=params.sam_crop_n_points_downscale_factor,
                sam_min_mask_region_area=params.sam_min_mask_region_area,
                selection_mode=params.selection_mode,
                selection_prompt=params.selection_prompt,
            )
            func = getattr(self, func_name)
            result = dict(func(tmp_path, sub_params, **kwargs) or {})
            detections = []
            for det in result.get("detections") or []:
                item = dict(det)
                box = item.get("box")
                if isinstance(box, list) and len(box) == 4:
                    item["box"] = [
                        float(box[0]) + roi_x1,
                        float(box[1]) + roi_y1,
                        float(box[2]) + roi_x1,
                        float(box[3]) + roi_y1,
                    ]
                detections.append(item)
            result["image_path"] = str(image_path)
            result["detections"] = detections
            return result
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def predict(self, image_path: str, params: SamParams, *, stop_checker=None, log_fn=None) -> dict:
        if params.roi_box is not None:
            return self._run_with_roi("predict", image_path, params, stop_checker=stop_checker, log_fn=log_fn)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.path.isfile(params.sam_checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found: {params.sam_checkpoint}")
        predictor, _device = self.ensure_model_loaded(params, log_fn=log_fn)
        return _process_one_image_sam_only(
            image_path=image_path,
            params=params,
            predictor=predictor,
            device=_device,
            stop_checker=stop_checker,
            log_fn=log_fn,
        )

    def segment_boxes(
        self,
        image_path: str,
        params: SamParams,
        boxes: Sequence[dict[str, Any]],
        *,
        stop_checker=None,
        log_fn=None,
    ) -> dict:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.path.isfile(params.sam_checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found: {params.sam_checkpoint}")
        predictor, _device = self.ensure_model_loaded(params, log_fn=log_fn)
        return _segment_boxes_with_predictor(
            image_path=image_path,
            params=params,
            predictor=predictor,
            boxes=boxes,
            stop_checker=stop_checker,
            log_fn=log_fn,
            model_name="Sam",
        )


def _process_one_image_sam_only(
    *,
    image_path: str,
    params: SamParams,
    predictor: Any,
    device: str,
    stop_checker=None,
    log_fn=None,
) -> dict:
    import cv2
    import numpy as np

    from sam.runtime import (
        _mask_bbox,
        _mask_stats,
        ensure_dir,
        filter_small_components,
        make_sam_auto_mask_generator,
        overlay_mask,
        safe_basename,
        score_mask_darkness,
        score_mask_for_crack,
    )

    def _auto_mask_scale(height: int, width: int, *, max_side: int, max_pixels: int) -> float:
        scale = 1.0
        longest = max(int(height), int(width))
        pixels = max(1, int(height) * int(width))
        if longest > int(max_side):
            scale = min(scale, float(max_side) / float(longest))
        if pixels > int(max_pixels):
            scale = min(scale, (float(max_pixels) / float(pixels)) ** 0.5)
        return min(1.0, float(scale))

    def _resize_image(image: np.ndarray, scale: float) -> np.ndarray:
        if scale >= 0.999:
            return image
        target_w = max(1, int(round(image.shape[1] * scale)))
        target_h = max(1, int(round(image.shape[0] * scale)))
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def _resize_mask_to_image(mask01: np.ndarray) -> np.ndarray:
        if mask01.shape[:2] == (h_img, w_img):
            return mask01.astype(np.uint8)
        resized = cv2.resize(mask01.astype(np.uint8), (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        return (resized > 0).astype(np.uint8)

    def _is_memory_error(exc: Exception) -> bool:
        text = str(exc or "").lower()
        return any(token in text for token in ("invalid buffer size", "out of memory", "bad allocation", "insufficient memory"))

    def _mask_center_score(mask01: np.ndarray, image_shape: tuple[int, int]) -> float:
        ys, xs = np.where(mask01 > 0)
        if len(xs) == 0 or len(ys) == 0:
            return 0.0
        h_ref, w_ref = int(image_shape[0]), int(image_shape[1])
        cx = float(xs.mean())
        cy = float(ys.mean())
        nx = (cx - (w_ref / 2.0)) / max(w_ref / 2.0, 1.0)
        ny = (cy - (h_ref / 2.0)) / max(h_ref / 2.0, 1.0)
        dist = float((nx * nx + ny * ny) ** 0.5)
        return max(0.0, 1.0 - min(dist / 1.41421356237, 1.0))

    def _mask_edge_touches(mask01: np.ndarray) -> int:
        touches = 0
        if np.any(mask01[0, :] > 0):
            touches += 1
        if np.any(mask01[-1, :] > 0):
            touches += 1
        if np.any(mask01[:, 0] > 0):
            touches += 1
        if np.any(mask01[:, -1] > 0):
            touches += 1
        return touches

    def _build_detection(mask01: np.ndarray, *, label: str, score: float, index: int) -> dict[str, Any] | None:
        if int(np.count_nonzero(mask01)) == 0:
            return None
        bbox = _mask_bbox(mask01)
        if bbox is None:
            return None
        success, png_bytes = cv2.imencode(".png", mask01.astype(np.uint8) * 255)
        mask_b64 = base64.b64encode(png_bytes.tobytes()).decode("ascii") if success else None
        return {
            "label": label,
            "score": float(score),
            "box": [float(bbox[0]), float(bbox[1]), float(bbox[2] - 1), float(bbox[3] - 1)],
            "mask_b64": mask_b64,
            "model_name": "SamOnlyAuto",
            "mask_index": int(index),
        }

    def _build_auto_generator(profile_name: str, *, safe_mode: bool):
        upper_profile = str(profile_name or "ULTRA").strip().upper()
        overrides = {
            "points_per_side": params.sam_points_per_side,
            "points_per_batch": params.sam_points_per_batch,
            "pred_iou_thresh": params.sam_pred_iou_thresh,
            "stability_score_thresh": params.sam_stability_score_thresh,
            "stability_score_offset": params.sam_stability_score_offset,
            "box_nms_thresh": params.sam_box_nms_thresh,
            "crop_n_layers": params.sam_crop_n_layers,
            "crop_overlap_ratio": params.sam_crop_overlap_ratio,
            "crop_nms_thresh": params.sam_crop_nms_thresh,
            "crop_n_points_downscale_factor": params.sam_crop_n_points_downscale_factor,
            "min_mask_region_area": params.sam_min_mask_region_area,
        }
        if safe_mode:
            if upper_profile == "QUALITY":
                overrides.update(
                    {
                        "points_per_side": 16 if int(params.sam_points_per_side) < 0 else min(int(params.sam_points_per_side), 16),
                        "points_per_batch": 32 if int(params.sam_points_per_batch) < 0 else min(int(params.sam_points_per_batch), 32),
                        "crop_n_layers": 0 if int(params.sam_crop_n_layers) < 0 else min(int(params.sam_crop_n_layers), 0),
                        "crop_n_points_downscale_factor": 2
                        if int(params.sam_crop_n_points_downscale_factor) < 0
                        else max(int(params.sam_crop_n_points_downscale_factor), 2),
                        "min_mask_region_area": 128
                        if int(params.sam_min_mask_region_area) < 0
                        else max(int(params.sam_min_mask_region_area), 128),
                    }
                )
            else:
                overrides.update(
                    {
                        "points_per_side": 8 if int(params.sam_points_per_side) < 0 else min(int(params.sam_points_per_side), 8),
                        "points_per_batch": 16 if int(params.sam_points_per_batch) < 0 else min(int(params.sam_points_per_batch), 16),
                        "crop_n_layers": 0 if int(params.sam_crop_n_layers) < 0 else min(int(params.sam_crop_n_layers), 0),
                        "crop_n_points_downscale_factor": 4
                        if int(params.sam_crop_n_points_downscale_factor) < 0
                        else max(int(params.sam_crop_n_points_downscale_factor), 4),
                        "min_mask_region_area": 256
                        if int(params.sam_min_mask_region_area) < 0
                        else max(int(params.sam_min_mask_region_area), 256),
                    }
                )
        return make_sam_auto_mask_generator(
            getattr(predictor, "model"),
            profile=profile_name,
            **overrides,
        )

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = bgr.shape[:2]
    ensure_dir(params.output_dir)
    base = safe_basename(image_path)
    overlay_path = os.path.join(params.output_dir, f"{base}_sam_only_overlay.png")
    mask_path = os.path.join(params.output_dir, f"{base}_crack_mask.png")
    if stop_checker is not None and stop_checker():
        return {"stopped": True}

    requested_profile = str(params.sam_auto_profile or "auto").strip().lower()
    if requested_profile and requested_profile != "auto":
        profile = requested_profile.upper()
    elif str(device).lower() == "mps":
        profile = "FAST"
    elif str(device).lower() == "cpu":
        profile = "QUALITY"
    else:
        profile = "ULTRA"
    if str(device).lower() == "mps" and profile == "QUALITY":
        work_scale = _auto_mask_scale(
            h_img,
            w_img,
            max_side=min(_SAM_AUTO_MAX_SIDE, _SAM_AUTO_MPS_QUALITY_MAX_SIDE),
            max_pixels=min(_SAM_AUTO_MAX_PIXELS, _SAM_AUTO_MPS_QUALITY_MAX_PIXELS),
        )
    else:
        work_scale = _auto_mask_scale(h_img, w_img, max_side=_SAM_AUTO_MAX_SIDE, max_pixels=_SAM_AUTO_MAX_PIXELS)
    work_bgr = _resize_image(bgr, work_scale)
    work_rgb = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB)
    h_work, w_work = work_bgr.shape[:2]
    preemptive_safe_mode = str(device).lower() == "mps" and profile in {"QUALITY", "ULTRA"} and (
        max(h_work, w_work) >= 1536 or (h_work * w_work) >= 2_000_000
    )
    auto_gen = _build_auto_generator(profile, safe_mode=preemptive_safe_mode)
    if log_fn is not None:
        log_fn(f"SAM only auto-mask mode: profile={profile}")
        if work_scale < 0.999:
            log_fn(f"SAM only auto-mask mode: resized {w_img}x{h_img} -> {w_work}x{h_work} for memory safety")
        if preemptive_safe_mode:
            log_fn("SAM only auto-mask mode: using adjusted settings for large image on MPS")
        log_fn("SAM only auto-mask mode: generating masks...")
    try:
        masks_info = auto_gen.generate(work_rgb)
    except RuntimeError as exc:
        if not _is_memory_error(exc):
            raise
        retry_scale = min(
            work_scale,
            _auto_mask_scale(h_img, w_img, max_side=_SAM_AUTO_RETRY_MAX_SIDE, max_pixels=_SAM_AUTO_RETRY_MAX_PIXELS),
        )
        work_scale = retry_scale
        work_bgr = _resize_image(bgr, work_scale)
        work_rgb = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB)
        h_work, w_work = work_bgr.shape[:2]
        if log_fn is not None:
            log_fn(
                "SAM auto-mask hit memory limits; retrying with safer settings "
                f"at {w_work}x{h_work}."
            )
        auto_gen = _build_auto_generator("FAST", safe_mode=True)
        masks_info = auto_gen.generate(work_rgb)
    if not masks_info:
        cv2.imwrite(overlay_path, bgr)
        cv2.imwrite(mask_path, np.zeros((h_img, w_img), dtype=np.uint8))
        return {"image_path": str(image_path), "overlay_path": overlay_path, "mask_path": mask_path, "output_dir": params.output_dir, "detections": []}
    if log_fn is not None:
        log_fn(f"SAM only auto-mask mode: generated {len(masks_info)} masks")

    gray = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2GRAY)
    task_group = str(params.task_group or "crack_only").strip().lower()
    selection_mode = str(params.selection_mode or "default").strip().lower()
    selection_prompt = str(params.selection_prompt or "").strip()
    detection_prefix = selection_prompt or ("DamageMask" if task_group == "more_damage" else "CrackMask")
    rng = np.random.default_rng(int(params.seed))
    disp = bgr.copy()
    merged = np.zeros((h_img, w_img), dtype=np.uint8)
    final_detections: list[dict[str, Any]] = []
    if task_group == "more_damage":
        ranked_masks: list[tuple[float, np.ndarray, dict[str, float], int]] = []
        for info in masks_info:
            if stop_checker is not None and stop_checker():
                return {"stopped": True}
            seg = info.get("segmentation")
            if seg is None:
                continue
            mask = seg.astype(np.uint8)
            if int(np.count_nonzero(mask)) <= 0:
                continue
            predicted_iou = float(info.get("predicted_iou", 0.0))
            stability = float(info.get("stability_score", 0.0))
            score = predicted_iou + 0.2 * stability
            stats = _mask_stats(mask, (h_work, w_work))
            if stats is None:
                continue
            edge_touches = _mask_edge_touches(mask)
            if selection_mode == "isolate":
                center_score = _mask_center_score(mask, (h_work, w_work))
                bbox_coverage = 0.5 * ((stats["bbox_w"] / max(float(w_work), 1.0)) + (stats["bbox_h"] / max(float(h_work), 1.0)))
                score += 2.0 * center_score
                score += 2.0 * min(bbox_coverage, 0.75)
                if 0.04 <= stats["image_ratio"] <= 0.75:
                    score += 2.0
                elif stats["image_ratio"] < 0.015:
                    score -= 2.0
                elif stats["image_ratio"] < 0.04:
                    score -= 0.5
                score -= 0.8 * float(edge_touches)
                if edge_touches >= 3:
                    score -= 2.5
                if stats["fill_ratio"] > 0.92 and edge_touches >= 1:
                    score -= 1.5
                if stats["image_ratio"] > 0.82:
                    score -= 3.5
            ranked_masks.append((score, mask, stats, edge_touches))
        ranked_masks.sort(key=lambda item: item[0], reverse=True)
        keep_count = max(1, int(params.more_damage_max_masks))
        if selection_mode == "isolate":
            top_score = ranked_masks[0][0] if ranked_masks else float("-inf")
            filtered = [
                item for item in ranked_masks
                if item[0] >= top_score - 2.0 and (item[2]["image_ratio"] >= 0.03 or item[2]["bbox_h"] >= 0.18 * h_work)
            ]
            keep_count = min(max(2, keep_count), 6)
            kept_damage = filtered[:keep_count] if filtered else ranked_masks[:1]
        else:
            kept_damage = ranked_masks[:keep_count]
        if log_fn is not None:
            if selection_mode == "isolate":
                suffix = f" for prompt='{selection_prompt}'" if selection_prompt else ""
                log_fn(f"SAM only auto-mask mode: kept {len(kept_damage)} isolate-focused mask{suffix}")
            else:
                log_fn(f"SAM only auto-mask mode: kept {len(kept_damage)} damage masks")
        for det_index, (_score, chosen, _stats, _edge_touches) in enumerate(kept_damage, start=1):
            if params.invert_mask:
                chosen = (1 - chosen).astype(np.uint8)
            chosen = filter_small_components(chosen, int(params.sam_min_component_area))
            if int(params.sam_dilate_iters) > 0 and int(np.count_nonzero(chosen)) > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                chosen = cv2.dilate(chosen.astype(np.uint8), kernel, iterations=int(params.sam_dilate_iters)).astype(np.uint8)
            chosen = _resize_mask_to_image(chosen)
            if int(np.count_nonzero(chosen)) == 0:
                continue
            merged = np.maximum(merged, chosen)
            color = rng.integers(0, 255, (3,), dtype=np.uint8)
            disp = overlay_mask(disp, chosen, color=color, alpha=float(params.overlay_alpha))
            det = _build_detection(
                chosen,
                label=f"{detection_prefix} #{det_index}" if len(kept_damage) > 1 else detection_prefix,
                score=_score,
                index=det_index,
            )
            if det is not None:
                final_detections.append(det)
    else:
        ranked_masks: list[tuple[float, np.ndarray, dict[str, float], float, float]] = []
        for info in masks_info:
            if stop_checker is not None and stop_checker():
                return {"stopped": True}
            seg = info.get("segmentation")
            if seg is None:
                continue
            mask = seg.astype(np.uint8)
            stats = _mask_stats(mask, (h_work, w_work))
            if stats is None:
                continue
            stability = float(info.get("stability_score", 0.0))
            shape_score = score_mask_for_crack(mask, stability, (h_work, w_work))
            darkness_score = score_mask_darkness(mask, gray)
            ranked_masks.append((shape_score + 0.35 * darkness_score, mask, stats, stability, darkness_score))
        ranked_masks.sort(key=lambda x: x[0], reverse=True)

        kept = []
        for score, mask, stats, stability, darkness_score in ranked_masks:
            if stats["image_ratio"] > 0.12:
                continue
            if stats["fill_ratio"] > 0.55 and stats["image_ratio"] > 0.02:
                continue
            if stats["elongation"] < 1.8 and stats["thinness"] < 2.2:
                continue
            if darkness_score < 6.0 and stats["elongation"] < 3.0:
                continue
            if darkness_score < 3.0:
                continue
            kept.append((score, mask, stats, stability, darkness_score))
            if len(kept) >= 12:
                break

        if log_fn is not None:
            log_fn(f"SAM only auto-mask mode: kept {len(kept)} crack-like masks")

        for det_index, (_score, chosen, _stats, _stability, _darkness) in enumerate(kept, start=1):
            if params.invert_mask:
                chosen = (1 - chosen).astype(np.uint8)
            chosen = filter_small_components(chosen, int(params.sam_min_component_area))
            if int(params.sam_dilate_iters) > 0 and int(np.count_nonzero(chosen)) > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                chosen = cv2.dilate(chosen.astype(np.uint8), kernel, iterations=int(params.sam_dilate_iters)).astype(np.uint8)
            chosen = _resize_mask_to_image(chosen)
            if int(np.count_nonzero(chosen)) == 0:
                continue
            merged = np.maximum(merged, chosen)
            color = rng.integers(0, 255, (3,), dtype=np.uint8)
            disp = overlay_mask(disp, chosen, color=color, alpha=float(params.overlay_alpha))
            det = _build_detection(
                chosen,
                label=f"{detection_prefix} #{det_index}" if len(kept) > 1 else detection_prefix,
                score=_score,
                index=det_index,
            )
            if det is not None:
                final_detections.append(det)

    if int(np.count_nonzero(merged)) == 0:
        cv2.imwrite(overlay_path, bgr)
        cv2.imwrite(mask_path, np.zeros((h_img, w_img), dtype=np.uint8))
        return {"image_path": str(image_path), "overlay_path": overlay_path, "mask_path": mask_path, "output_dir": params.output_dir, "detections": []}

    cv2.imwrite(overlay_path, disp)
    cv2.imwrite(mask_path, merged * 255)
    return {
        "image_path": str(image_path),
        "overlay_path": overlay_path,
        "mask_path": mask_path,
        "output_dir": params.output_dir,
        "masks_saved": len(final_detections),
        "detections": final_detections,
    }


def _segment_boxes_with_predictor(
    *,
    image_path: str,
    params: SamParams,
    predictor: Any,
    boxes: Sequence[dict[str, Any]],
    stop_checker=None,
    log_fn=None,
    model_name: str,
) -> dict:
    import cv2
    import numpy as np

    from sam.runtime import ensure_dir, filter_small_components, overlay_mask, safe_basename, select_sam_mask

    ensure_dir(params.output_dir)
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = bgr.shape[:2]
    predictor.set_image(rgb)
    disp = bgr.copy()
    merged = np.zeros((h_img, w_img), dtype=np.uint8)
    final_dets = []
    rng = np.random.default_rng(int(params.seed))

    for entry in boxes:
        if stop_checker is not None and stop_checker():
            return {"stopped": True}
        box = entry.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        label = str(entry.get("label") or "object")
        score = float(entry.get("score") or 0.0)
        x1 = float(max(0, min(w_img - 1, x1)))
        x2 = float(max(0, min(w_img - 1, x2)))
        y1 = float(max(0, min(h_img - 1, y1)))
        y2 = float(max(0, min(h_img - 1, y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        x1i = max(0, min(w_img, int(np.floor(x1))))
        y1i = max(0, min(h_img, int(np.floor(y1))))
        x2i = max(0, min(w_img, int(np.ceil(x2)) + 1))
        y2i = max(0, min(h_img, int(np.ceil(y2)) + 1))
        if x2i <= x1i or y2i <= y1i:
            continue
        cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
        cv2.putText(
            disp,
            label,
            (int(x1), int(max(0, y1 - 5))),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 200, 255),
            2,
        )
        input_box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)
        if masks is None or len(masks) == 0:
            continue
        prefer_crack = str(params.task_group or "crack_only").strip().lower() != "more_damage" and ("crack" in label.lower())
        chosen = select_sam_mask(masks, scores, (h_img, w_img), prefer_crack=prefer_crack).astype(np.uint8)
        if params.invert_mask:
            chosen = (1 - chosen).astype(np.uint8)
        clip = np.zeros_like(chosen, dtype=np.uint8)
        clip[y1i:y2i, x1i:x2i] = chosen[y1i:y2i, x1i:x2i]
        chosen = filter_small_components(clip, int(params.sam_min_component_area))
        if int(params.sam_dilate_iters) > 0 and int(np.count_nonzero(chosen)) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            chosen = cv2.dilate(chosen.astype(np.uint8), kernel, iterations=int(params.sam_dilate_iters)).astype(np.uint8)
        merged = np.maximum(merged, chosen)
        color = rng.integers(0, 255, (3,), dtype=np.uint8)
        disp = overlay_mask(disp, chosen, color=color, alpha=float(params.overlay_alpha))
        success, png_bytes = cv2.imencode(".png", chosen * 255)
        mask_b64 = base64.b64encode(png_bytes.tobytes()).decode("ascii") if success else None
        final_dets.append(
            {
                "label": label,
                "score": score,
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "mask_b64": mask_b64,
                "model_name": model_name,
            }
        )

    base = safe_basename(image_path)
    overlay_path = os.path.join(params.output_dir, f"{base}_{model_name.lower()}_boxes_overlay.png")
    mask_path = os.path.join(params.output_dir, f"{base}_{model_name.lower()}_boxes_mask.png")
    cv2.imwrite(overlay_path, disp)
    cv2.imwrite(mask_path, merged * 255)
    if log_fn is not None:
        masks_saved = 1 if int(np.count_nonzero(merged)) > 0 else 0
        log_fn(f"SAM box prompting done. masks_saved={masks_saved}")
    return {
        "image_path": str(image_path),
        "overlay_path": overlay_path,
        "mask_path": mask_path,
        "output_dir": params.output_dir,
        "masks_saved": 1 if int(np.count_nonzero(merged)) > 0 else 0,
        "detections": final_dets,
    }
