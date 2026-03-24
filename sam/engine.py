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
        needs_reload = (
            self._predictor is None
            or self._sam_checkpoint != params.sam_checkpoint
            or self._sam_model_type != params.sam_model_type
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

    if str(device).lower() == "mps":
        profile = "FAST"
    elif str(device).lower() == "cpu":
        profile = "QUALITY"
    else:
        profile = "ULTRA"
    auto_gen = make_sam_auto_mask_generator(getattr(predictor, "model"), profile=profile)
    if log_fn is not None:
        log_fn(f"SAM only auto-mask mode: profile={profile}")
    masks_info = auto_gen.generate(rgb)
    if not masks_info:
        cv2.imwrite(overlay_path, bgr)
        cv2.imwrite(mask_path, np.zeros((h_img, w_img), dtype=np.uint8))
        return {"image_path": str(image_path), "overlay_path": overlay_path, "mask_path": mask_path, "output_dir": params.output_dir, "detections": []}
    if log_fn is not None:
        log_fn(f"SAM only auto-mask mode: generated {len(masks_info)} masks")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rng = np.random.default_rng(int(params.seed))
    disp = bgr.copy()
    merged = np.zeros((h_img, w_img), dtype=np.uint8)
    ranked_masks: list[tuple[float, np.ndarray, dict[str, float], float, float]] = []
    for info in masks_info:
        if stop_checker is not None and stop_checker():
            return {"stopped": True}
        seg = info.get("segmentation")
        if seg is None:
            continue
        mask = seg.astype(np.uint8)
        stats = _mask_stats(mask, (h_img, w_img))
        if stats is None:
            continue
        stability = float(info.get("stability_score", 0.0))
        shape_score = score_mask_for_crack(mask, stability, (h_img, w_img))
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

    for _score, chosen, _stats, _stability, _darkness in kept:
        if params.invert_mask:
            chosen = (1 - chosen).astype(np.uint8)
        chosen = filter_small_components(chosen, int(params.sam_min_component_area))
        if int(params.sam_dilate_iters) > 0 and int(np.count_nonzero(chosen)) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            chosen = cv2.dilate(chosen.astype(np.uint8), kernel, iterations=int(params.sam_dilate_iters)).astype(np.uint8)
        if int(np.count_nonzero(chosen)) == 0:
            continue
        merged = np.maximum(merged, chosen)
        color = rng.integers(0, 255, (3,), dtype=np.uint8)
        disp = overlay_mask(disp, chosen, color=color, alpha=float(params.overlay_alpha))

    if int(np.count_nonzero(merged)) == 0:
        cv2.imwrite(overlay_path, bgr)
        cv2.imwrite(mask_path, np.zeros((h_img, w_img), dtype=np.uint8))
        return {"image_path": str(image_path), "overlay_path": overlay_path, "mask_path": mask_path, "output_dir": params.output_dir, "detections": []}

    success, png_bytes = cv2.imencode(".png", merged * 255)
    mask_b64 = base64.b64encode(png_bytes.tobytes()).decode("ascii") if success else None
    bbox = _mask_bbox(merged) or (0, 0, w_img, h_img)
    detections = [
        {
            "label": "CrackMask",
            "score": 1.0,
            "box": [float(bbox[0]), float(bbox[1]), float(bbox[2] - 1), float(bbox[3] - 1)],
            "mask_b64": mask_b64,
            "model_name": "SamOnlyAuto",
        }
    ]
    cv2.imwrite(overlay_path, disp)
    cv2.imwrite(mask_path, merged * 255)
    return {
        "image_path": str(image_path),
        "overlay_path": overlay_path,
        "mask_path": mask_path,
        "output_dir": params.output_dir,
        "masks_saved": 1,
        "detections": detections,
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
            f"{label} {score:.2f}",
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
        prefer_crack = "crack" in label.lower()
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
