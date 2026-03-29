from __future__ import annotations

import base64
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from torch_runtime import describe_device_fallback, select_device_str


@dataclass(frozen=True)
class SamFinetuneParams:
    sam_checkpoint: str
    delta_type: str
    delta_checkpoint: str = "auto"
    sam_model_type: str = "auto"
    middle_dim: int = 32
    scaling_factor: float = 0.2
    rank: int = 4
    invert_mask: bool = False
    sam_min_component_area: int = 0
    sam_dilate_iters: int = 0
    seed: int = 1337
    overlay_alpha: float = 0.45
    device: str = "auto"
    output_dir: str = "results_sam_finetune"
    predict_mode: str = "auto"
    tile_size: int = -1
    tile_overlap: int = -1
    threshold: str = "auto"
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


class SamFinetuneRunner:
    def __init__(self) -> None:
        self._device: str | None = None
        self._sam_checkpoint: str | None = None
        self._sam_model_type: str | None = None
        self._delta_sig: tuple | None = None
        self._resolved_delta_checkpoint: str | None = None
        self._predictor: Any | None = None

    def _ensure_import_paths(self) -> None:
        here = Path(__file__).resolve()
        repo_root = here.parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

    def ensure_model_loaded(self, params: SamFinetuneParams, *, log_fn=None) -> tuple[Any, str]:
        self._ensure_import_paths()

        from sam.runtime import load_sam_model
        from segment_anything import SamPredictor
        from sam_finetune.runtime import apply_delta_to_sam, infer_delta_type_from_path, resolve_best_delta_checkpoint

        device = select_device_str(params.device)
        fallback = describe_device_fallback(params.device, device)
        if fallback is not None and log_fn is not None:
            log_fn(fallback)
        delta_type = str(params.delta_type or "").strip().lower()
        delta_path = resolve_best_delta_checkpoint(delta_type, str(params.delta_checkpoint or "auto"))
        inferred = infer_delta_type_from_path(delta_path)
        if inferred is not None and inferred != delta_type:
            raise ValueError(f"Delta checkpoint mismatch: checkpoint looks like {inferred}, expected {delta_type}.")
        delta_sig = (
            delta_type,
            delta_path,
            int(params.middle_dim),
            float(params.scaling_factor),
            int(params.rank),
        )
        requested_model_type = str(params.sam_model_type or "auto").strip().lower()
        needs_reload = (
            self._predictor is None
            or self._sam_checkpoint != params.sam_checkpoint
            or (requested_model_type != "auto" and self._sam_model_type != requested_model_type)
            or self._delta_sig != delta_sig
            or self._device != device
        )
        if not needs_reload:
            return self._predictor, device
        if log_fn is not None:
            log_fn("Loading SAM checkpoint...")
        sam_model, used_model_type = load_sam_model(params.sam_checkpoint, params.sam_model_type)
        if log_fn is not None:
            log_fn(f"Applying delta to SAM... type={delta_type} ckpt={delta_path}")
        apply_delta_to_sam(
            sam=sam_model,
            delta_type=delta_type,
            delta_ckpt_path=str(delta_path),
            middle_dim=int(params.middle_dim),
            scaling_factor=float(params.scaling_factor),
            rank=int(params.rank),
        )
        sam_model.to(device=device)
        self._predictor = SamPredictor(sam_model)
        self._sam_checkpoint = params.sam_checkpoint
        self._sam_model_type = used_model_type
        self._delta_sig = delta_sig
        self._resolved_delta_checkpoint = str(delta_path)
        self._device = device
        if log_fn is not None:
            log_fn(f"SAM finetune ready (type={used_model_type}, device={device}).")
        return self._predictor, device

    def _predict_score_map_tiled(self, predictor, rgb_image, *, tile_size: int, tile_overlap: int):
        from sam_finetune.tiled_inference import predictor_tile_mask_score, tiled_score_map

        return tiled_score_map(
            rgb_image,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            predict_tile_mask_score=lambda tile: predictor_tile_mask_score(predictor, tile),
        )

    def _run_with_roi(self, func_name: str, image_path: str, params: SamFinetuneParams, **kwargs) -> dict:
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
            sub_params = SamFinetuneParams(
                sam_checkpoint=params.sam_checkpoint,
                delta_type=params.delta_type,
                delta_checkpoint=params.delta_checkpoint,
                sam_model_type=params.sam_model_type,
                middle_dim=params.middle_dim,
                scaling_factor=params.scaling_factor,
                rank=params.rank,
                invert_mask=params.invert_mask,
                sam_min_component_area=params.sam_min_component_area,
                sam_dilate_iters=params.sam_dilate_iters,
                seed=params.seed,
                overlay_alpha=params.overlay_alpha,
                device=params.device,
                output_dir=params.output_dir,
                predict_mode=params.predict_mode,
                tile_size=params.tile_size,
                tile_overlap=params.tile_overlap,
                threshold=params.threshold,
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

    def predict(self, image_path: str, params: SamFinetuneParams, *, stop_checker=None, log_fn=None) -> dict:
        import cv2
        import numpy as np

        from sam.runtime import ensure_dir, filter_small_components, overlay_mask, safe_basename
        from sam_finetune.runtime import (
            resolve_predict_mode,
            resolve_predict_threshold,
            resolve_tile_settings,
        )

        if params.roi_box is not None:
            return self._run_with_roi("predict", image_path, params, stop_checker=stop_checker, log_fn=log_fn)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.path.isfile(params.sam_checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found: {params.sam_checkpoint}")
        predictor, _device = self.ensure_model_loaded(params, log_fn=log_fn)
        delta_path = self._resolved_delta_checkpoint
        predict_mode = resolve_predict_mode(delta_path, params.predict_mode)
        threshold = resolve_predict_threshold(delta_path, params.threshold)
        tile_size, tile_overlap = resolve_tile_settings(delta_path, params.tile_size, params.tile_overlap)
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
        h_img, w_img = bgr.shape[:2]
        ensure_dir(params.output_dir)
        base = safe_basename(image_path)
        overlay_path = os.path.join(params.output_dir, f"{base}_sam_only_overlay.png")
        mask_path = os.path.join(params.output_dir, f"{base}_crack_mask.png")
        if stop_checker is not None and stop_checker():
            return {"stopped": True}
        if log_fn is not None:
            if predict_mode == "tile_full_box":
                log_fn(
                    f"SAM finetune tiled predict: tile_size={tile_size} tile_overlap={tile_overlap} threshold={threshold:.4f}"
                )
            else:
                log_fn(f"SAM finetune legacy predict: threshold={threshold:.4f}")

        if predict_mode == "legacy_full_box":
            predictor.set_image(rgb)
            full_box = np.array([[0.0, 0.0, float(w_img - 1), float(h_img - 1)]], dtype=np.float32)
            masks, scores, _ = predictor.predict(box=full_box, multimask_output=True)
            if masks is None or len(masks) == 0:
                chosen = np.zeros((h_img, w_img), dtype=np.uint8)
                score = 0.0
            else:
                idx = int(np.argmax(scores))
                prob_map = masks[idx].astype(np.float32)
                chosen = (prob_map >= float(threshold)).astype(np.uint8)
                score = float(scores[idx])
        else:
            score_map = self._predict_score_map_tiled(
                predictor,
                rgb,
                tile_size=int(tile_size),
                tile_overlap=int(tile_overlap),
            )
            chosen = (score_map >= float(threshold)).astype(np.uint8)
            score = float(score_map[chosen > 0].mean()) if int(np.count_nonzero(chosen)) > 0 else float(score_map.max())

        if params.invert_mask:
            chosen = (1 - chosen).astype(np.uint8)
        chosen = filter_small_components(chosen, int(params.sam_min_component_area))
        if int(params.sam_dilate_iters) > 0 and int(np.count_nonzero(chosen)) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            chosen = cv2.dilate(chosen.astype(np.uint8), kernel, iterations=int(params.sam_dilate_iters)).astype(np.uint8)
        rng = np.random.default_rng(int(params.seed))
        disp = bgr.copy()
        color = rng.integers(0, 255, (3,), dtype=np.uint8)
        disp = overlay_mask(disp, chosen, color=color, alpha=float(params.overlay_alpha))
        cv2.imwrite(overlay_path, disp)
        cv2.imwrite(mask_path, chosen * 255)
        success, png_bytes = cv2.imencode(".png", chosen * 255)
        mask_b64 = base64.b64encode(png_bytes.tobytes()).decode("ascii") if success else None
        detections = [
            {
                "label": "Mask",
                "score": float(score),
                "box": [0.0, 0.0, float(w_img - 1), float(h_img - 1)],
                "mask_b64": mask_b64,
                "model_name": "SamFinetune",
            }
        ]
        return {
            "image_path": str(image_path),
            "overlay_path": overlay_path,
            "mask_path": mask_path,
            "output_dir": params.output_dir,
            "masks_saved": 1 if int(np.count_nonzero(chosen)) > 0 else 0,
            "detections": detections,
        }

    def segment_boxes(
        self,
        image_path: str,
        params: SamFinetuneParams,
        boxes: Sequence[dict[str, Any]],
        *,
        stop_checker=None,
        log_fn=None,
    ) -> dict:
        from sam.engine import _segment_boxes_with_predictor, SamParams

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.path.isfile(params.sam_checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found: {params.sam_checkpoint}")
        predictor, _device = self.ensure_model_loaded(params, log_fn=log_fn)
        base_params = SamParams(
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
        )
        return _segment_boxes_with_predictor(
            image_path=image_path,
            params=base_params,
            predictor=predictor,
            boxes=boxes,
            stop_checker=stop_checker,
            log_fn=log_fn,
            model_name="SamFinetune",
        )
