from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UnetParams:
    model_path: str
    output_dir: str
    threshold: float = 0.5
    apply_postprocessing: bool = True
    mode: str = "tile"  # tile|letterbox|resize
    input_size: int = 256
    tile_overlap: int = 0  # 0 means recommended (input_size//2)
    tile_batch_size: int = 4
    device: str = "auto"  # auto|cpu|mps|cuda
    roi_box: tuple[int, int, int, int] | None = None


class UnetRunner:
    def __init__(self) -> None:
        self._loaded_weights: str | None = None
        self._device: str | None = None
        self._model: Any | None = None

    def _ensure_import_paths(self) -> None:
        here = Path(__file__).resolve()
        repo_root = here.parents[3]
        unet_root = repo_root / "unet"
        for p in (str(repo_root), str(unet_root)):
            if p not in sys.path:
                sys.path.insert(0, p)

    def ensure_model_loaded(self, params: UnetParams, *, log_fn=None) -> None:
        self._ensure_import_paths()
        import torch

        from device_utils import select_device_str
        import segmentation_models_pytorch as smp

        if not os.path.isfile(params.model_path):
            raise FileNotFoundError(f"Model not found: {params.model_path}")

        device = select_device_str(params.device, torch=torch)
        needs_reload = (self._model is None) or (self._loaded_weights != params.model_path) or (self._device != device)
        if not needs_reload:
            return

        if log_fn is not None:
            log_fn(f"Loading UNet weights... ({params.model_path})")
        
        # Initialize SMP UNet
        # Try loading with EfficientNet-B4 (default in training) first, then fallback to ConvNeXt-Tiny
        encoders_to_try = ["efficientnet-b4", "tu-convnext_tiny"]
        
        last_exception = None
        m = None
        
        state = torch.load(params.model_path, map_location=device, weights_only=False)
        
        for enc_name in encoders_to_try:
            try:
                if log_fn: log_fn(f"Attempting to load model with encoder: {enc_name}...")
                m = smp.Unet(
                    encoder_name=enc_name,
                    encoder_weights=None, 
                    in_channels=3, 
                    classes=1,
                    decoder_attention_type="scse"
                )
                m.load_state_dict(state)
                if log_fn: log_fn(f"Successfully loaded model with encoder: {enc_name}")
                break
            except Exception as e:
                last_exception = e
                m = None
        
        if m is None:
             if log_fn: log_fn(f"Error loading model: {last_exception}. Ensure architecture matches training config.")
             raise last_exception

        m = m.to(device)
        m.eval()
        self._model = m
        self._loaded_weights = params.model_path
        self._device = device
        if log_fn is not None:
            log_fn(f"UNet ready (device={device}).")

    def run(self, image_path: str, params: UnetParams, *, stop_checker=None, log_fn=None) -> dict:
        self._ensure_import_paths()

        import torch

        from device_utils import select_device_str
        from predict_lib.core import predict_image

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        os.makedirs(params.output_dir, exist_ok=True)

        device = select_device_str(params.device, torch=torch)
        if str(params.device or "").strip().lower() == "cuda" and device != "cuda" and log_fn is not None:
            log_fn("WARN: CUDA not available, falling back to CPU.")
        if str(params.device or "").strip().lower() == "mps" and device != "mps" and log_fn is not None:
            log_fn("WARN: MPS not available, falling back to CPU.")

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
        return dict(details)

    def run_rois(
        self,
        image_path: str,
        params: UnetParams,
        rois: list[tuple[int, int, int, int]],
        *,
        stop_checker=None,
        log_fn=None,
    ) -> dict:
        # Runs UNet on each ROI and stitches results
        import numpy as np
        import cv2
        import torch
        from PIL import Image
        from predict_lib.inference import predict_probabilities
        from predict_lib.postprocess import binarize_prediction, postprocess_binary_mask
        from device_utils import select_device_str
        
        self.ensure_model_loaded(params, log_fn=log_fn)
        assert self._model is not None
        device = self._device
        
        bgr = cv2.imread(image_path)
        if bgr is None:
             raise FileNotFoundError(f"Image not found {image_path}")
        # Convert BGR to RGB for PIL
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img = rgb.shape[:2]
        
        final_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        
        os.makedirs(params.output_dir, exist_ok=True)
        base = Path(image_path).stem
        
        roi_count = len(rois)
        if log_fn:
            log_fn(f"Running UNet on {roi_count} ROIs...")
            
        for i, roi in enumerate(rois):
            if stop_checker and stop_checker():
                return {"stopped": True}

            l, t, r, b = roi
            # Sanity check
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
                log_fn(f"UNet ROI {i+1}/{roi_count}: {l},{t},{r},{b}")

            # Predict
            overlap = (params.input_size // 2) if int(params.tile_overlap) == 0 else int(params.tile_overlap)
            
            prob_map = predict_probabilities(
                self._model,
                crop_pil,
                device,
                mode=params.mode,
                input_size=params.input_size,
                tile_overlap=overlap,
                tile_batch_size=params.tile_batch_size,
                stop_checker=stop_checker
            )
            
            mask_bool = binarize_prediction(prob_map, params.threshold)
            mask_bool = postprocess_binary_mask(mask_bool, params.apply_postprocessing)
            
            # Merge
            roi_mask_uint8 = (mask_bool.astype(np.uint8) * 255)
            # Ensure shape match (probability map matches input crop size)
            if roi_mask_uint8.shape != (b-t, r-l):
                 roi_mask_uint8 = cv2.resize(roi_mask_uint8, (r-l, b-t), interpolation=cv2.INTER_NEAREST)
                 
            final_mask[t:b, l:r] = np.maximum(final_mask[t:b, l:r], roi_mask_uint8)
            
        # Save final merged mask
        mask_path = os.path.join(params.output_dir, f"{base}_unet_rois.png")
        overlay_path = os.path.join(params.output_dir, f"{base}_unet_rois_overlay.png")
        cv2.imwrite(mask_path, final_mask)
        
        # Create overlay
        alpha = 0.5
        overlay = bgr.copy()
        # Red overlay
        bool_mask = final_mask > 0
        overlay[bool_mask, 2] = (alpha * overlay[bool_mask, 2] + (1 - alpha) * 255).astype(np.uint8)
        # We can just blend everything properly if needed, but simple red channel boost is ok or standard blend
        # Standard blend:
        # BGR red is (0, 0, 255)
        # overlay[bool_mask] = overlay[bool_mask] * alpha + red * (1-alpha)
        
        cv2.imwrite(overlay_path, overlay)
        
        return {
            "mask_path": mask_path,
            "overlay_path": overlay_path,
            "output_dir": params.output_dir
        }
