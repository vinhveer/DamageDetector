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
        repo_root = here.parents[1]
        unet_root = repo_root / "unet"
        for p in (str(repo_root), str(unet_root)):
            if p not in sys.path:
                sys.path.insert(0, p)

    def run(self, image_path: str, params: UnetParams, *, stop_checker=None, log_fn=None) -> dict:
        self._ensure_import_paths()

        import torch

        from device_utils import select_device_str
        from predict_lib.core import predict_image
        from unet.unet_model import UNet

        if not os.path.isfile(params.model_path):
            raise FileNotFoundError(f"Model not found: {params.model_path}")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        os.makedirs(params.output_dir, exist_ok=True)

        device = select_device_str(params.device, torch=torch)
        if str(params.device or "").strip().lower() == "cuda" and device != "cuda" and log_fn is not None:
            log_fn("WARN: CUDA not available, falling back to CPU.")
        if str(params.device or "").strip().lower() == "mps" and device != "mps" and log_fn is not None:
            log_fn("WARN: MPS not available, falling back to CPU.")

        needs_reload = (self._model is None) or (self._loaded_weights != params.model_path) or (self._device != device)
        if needs_reload:
            if log_fn is not None:
                log_fn(f"Loading UNet weights... ({params.model_path})")
            m = UNet(in_channels=3, out_channels=1)
            state = torch.load(params.model_path, map_location=device)
            m.load_state_dict(state)
            m = m.to(device)
            m.eval()
            self._model = m
            self._loaded_weights = params.model_path
            self._device = device
            if log_fn is not None:
                log_fn(f"UNet ready (device={device}).")

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
