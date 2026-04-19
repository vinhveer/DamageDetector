from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch_runtime import describe_device_fallback, select_device_str


@dataclass(frozen=True)
class UnetParams:
    model_path: str
    output_dir: str
    threshold: float = 0.5
    apply_postprocessing: bool = True
    mode: str = "tile"
    input_size: int = 512
    tile_overlap: int = 0
    tile_batch_size: int = 4
    device: str = "auto"
    roi_box: tuple[int, int, int, int] | None = None
    task_group: str = "crack_only"


class UnetRunner:
    def __init__(self) -> None:
        self._loaded_weights: str | None = None
        self._device: str | None = None
        self._model: Any | None = None

    def ensure_model_loaded(self, params: UnetParams, *, log_fn=None) -> None:
        from .model_io import load_model_from_checkpoint, load_training_config_from_path

        if not os.path.isfile(params.model_path):
            raise FileNotFoundError(f"Model not found: {params.model_path}")
        device = select_device_str(params.device)
        needs_reload = (self._model is None) or (self._loaded_weights != params.model_path) or (self._device != device)
        if not needs_reload:
            return
        if log_fn is not None:
            log_fn(f"Loading UNet weights... ({params.model_path})")
        model, model_config = load_model_from_checkpoint(params.model_path, device)
        train_config = load_training_config_from_path(params.model_path)
        self._model = model
        self._loaded_weights = params.model_path
        self._device = device
        if log_fn is not None:
            msg = (
                f"UNet ready (device={device}, arch={model_config.get('arch')}, "
                f"encoder={model_config.get('encoder_name')})."
            )
            if train_config:
                args = train_config.get("args") or {}
                resolved = train_config.get("resolved") or {}
                msg += (
                    f" Trained with input_size={args.get('input_size')} | "
                    f"train={resolved.get('train_preprocess')} | val={resolved.get('val_preprocess')}."
                )
            log_fn(msg)

    def run(self, image_path: str, params: UnetParams, *, stop_checker=None, log_fn=None) -> dict:
        from .predict_lib.core import predict_image

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        os.makedirs(params.output_dir, exist_ok=True)
        device = select_device_str(params.device)
        fallback = describe_device_fallback(params.device, device)
        if fallback is not None and log_fn is not None:
            log_fn(fallback)
        self.ensure_model_loaded(params, log_fn=log_fn)
        assert self._model is not None
        overlap = (params.input_size // 2) if int(params.tile_overlap) == 0 else int(params.tile_overlap)
        if log_fn is not None:
            log_fn("Running UNet prediction...")
        details = predict_image(
            self._model,
            image_path,
            device,
            threshold=float(params.threshold),
            output_dir=params.output_dir,
            apply_postprocessing=bool(params.apply_postprocessing),
            return_details=True,
            mode=str(params.mode),
            input_size=int(params.input_size),
            tile_overlap=overlap,
            tile_batch_size=int(params.tile_batch_size),
            roi_box=params.roi_box,
            stop_checker=stop_checker,
        )
        if log_fn is not None:
            log_fn("UNet finished.")
        res = dict(details)
        res["image_path"] = str(image_path)
        return res

    def run_rois(
        self,
        image_path: str,
        params: UnetParams,
        rois: list[tuple[int, int, int, int]],
        *,
        stop_checker=None,
        log_fn=None,
    ) -> dict:
        import cv2
        import numpy as np
        from PIL import Image
        from .predict_lib.inference import predict_probabilities
        from .predict_lib.postprocess import binarize_prediction, postprocess_binary_mask

        self.ensure_model_loaded(params, log_fn=log_fn)
        assert self._model is not None
        device = self._device
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Image not found {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img = rgb.shape[:2]
        final_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        os.makedirs(params.output_dir, exist_ok=True)
        base = Path(image_path).stem
        if log_fn:
            log_fn(f"Running UNet on {len(rois)} ROIs...")
        for index, roi in enumerate(rois):
            if stop_checker and stop_checker():
                return {"stopped": True}
            l, t, r, b = roi
            l = max(0, min(w_img, l))
            t = max(0, min(h_img, t))
            r = max(0, min(w_img, r))
            b = max(0, min(h_img, b))
            if r <= l or b <= t:
                continue
            crop_arr = rgb[t:b, l:r]
            if crop_arr.size == 0:
                continue
            crop_pil = Image.fromarray(crop_arr)
            if log_fn:
                log_fn(f"UNet ROI {index+1}/{len(rois)}: {l},{t},{r},{b}")
            overlap = (params.input_size // 2) if int(params.tile_overlap) == 0 else int(params.tile_overlap)
            prob_map = predict_probabilities(
                self._model,
                crop_pil,
                device,
                mode=params.mode,
                input_size=params.input_size,
                tile_overlap=overlap,
                tile_batch_size=params.tile_batch_size,
                stop_checker=stop_checker,
            )
            mask_bool = binarize_prediction(prob_map, params.threshold)
            mask_bool = postprocess_binary_mask(mask_bool, params.apply_postprocessing)
            roi_mask_uint8 = mask_bool.astype(np.uint8) * 255
            if roi_mask_uint8.shape != (b - t, r - l):
                roi_mask_uint8 = cv2.resize(roi_mask_uint8, (r - l, b - t), interpolation=cv2.INTER_NEAREST)
            final_mask[t:b, l:r] = np.maximum(final_mask[t:b, l:r], roi_mask_uint8)
        mask_path = os.path.join(params.output_dir, f"{base}_unet_rois.png")
        overlay_path = os.path.join(params.output_dir, f"{base}_unet_rois_overlay.png")
        cv2.imwrite(mask_path, final_mask)
        alpha = 0.5
        overlay = bgr.copy()
        bool_mask = final_mask > 0
        overlay[bool_mask, 2] = (alpha * overlay[bool_mask, 2] + (1 - alpha) * 255).astype(np.uint8)
        cv2.imwrite(overlay_path, overlay)
        return {
            "mask_path": mask_path,
            "overlay_path": overlay_path,
            "output_dir": params.output_dir,
            "image_path": str(image_path),
        }
