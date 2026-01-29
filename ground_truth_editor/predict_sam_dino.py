from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class SamDinoParams:
    sam_checkpoint: str
    sam_model_type: str = "auto"  # auto|vit_b|vit_l|vit_h
    delta_type: str = "none"  # none|adapter|lora|both
    delta_checkpoint: str = "auto"  # "auto" or .pth path
    middle_dim: int = 32
    scaling_factor: float = 0.2
    rank: int = 4
    gdino_checkpoint: str = ""
    gdino_config_id: str = "auto"  # auto|IDEA-Research/grounding-dino-base|IDEA-Research/grounding-dino-tiny
    text_queries: Sequence[str] = ("crack",)
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    max_dets: int = 20
    overlay_alpha: float = 0.45
    invert_mask: bool = False
    sam_mask_threshold: float | None = None
    sam_min_component_area: int = 0
    sam_dilate_iters: int = 0
    seed: int = 1337
    device: str = "auto"  # auto|cpu|mps|cuda
    output_dir: str = "results_sam_dino"


class SamDinoRunner:
    def __init__(self) -> None:
        self._device: str | None = None
        self._sam_checkpoint: str | None = None
        self._sam_model_type: str | None = None
        self._delta_sig: tuple | None = None
        self._predictor: Any | None = None

        self._gdino_checkpoint: str | None = None
        self._gdino_config_id: str | None = None
        self._processor: Any | None = None
        self._gdino: Any | None = None

    def _infer_delta_type_from_path(self, path: str | None) -> str | None:
        if not path:
            return None
        name = os.path.basename(str(path)).lower()
        has_adapter = "adapter" in name
        has_lora = "lora" in name
        if has_adapter and has_lora:
            return "both"
        if has_adapter:
            return "adapter"
        if has_lora:
            return "lora"
        return None

    def _ensure_import_paths(self) -> None:
        here = Path(__file__).resolve()
        repo_root = here.parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

    def _load_gdino_state_dict(self, checkpoint_path: str) -> dict:
        import torch

        raw = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(raw, dict):
            if isinstance(raw.get("state_dict"), dict):
                state = raw["state_dict"]
            elif isinstance(raw.get("model"), dict):
                state = raw["model"]
            else:
                state = raw
        else:
            raise TypeError(f"Unsupported GroundingDINO checkpoint format: {type(raw)} ({checkpoint_path})")

        if not isinstance(state, dict):
            raise TypeError(f"GroundingDINO checkpoint state_dict invalid: {type(state)} ({checkpoint_path})")

        state = self._strip_prefix_if_present(state, "module.")
        state = self._strip_prefix_if_present(state, "model.")
        return state

    def _strip_prefix_if_present(self, state: dict, prefix: str) -> dict:
        keys = list(state.keys())
        if not keys:
            return state
        count = sum(1 for k in keys if str(k).startswith(prefix))
        if count == 0:
            return state
        if count >= int(len(keys) * 0.9):
            return {str(k)[len(prefix) :] if str(k).startswith(prefix) else k: v for k, v in state.items()}
        return state

    def _resolve_gdino_config_id(self, params: SamDinoParams) -> str:
        cfg = str(params.gdino_config_id or "").strip()
        if not cfg or cfg.lower() == "auto":
            name = os.path.basename(str(params.gdino_checkpoint or "")).lower()
            if "swint" in name or "tiny" in name:
                return "IDEA-Research/grounding-dino-tiny"
            return "IDEA-Research/grounding-dino-base"
        return cfg

    def _ensure_models(self, params: SamDinoParams, *, log_fn=None) -> tuple[Any, Any, Any, str]:
        self._ensure_import_paths()
        import torch

        from device_utils import select_device_str
        from sam_dino.pipeline import apply_delta_to_sam, load_sam_model, resolve_best_delta_checkpoint
        from segment_anything import SamPredictor  # type: ignore
        from transformers import AutoProcessor, GroundingDinoConfig, GroundingDinoForObjectDetection  # type: ignore

        device = select_device_str(params.device, torch=torch)
        if str(params.device or "").strip().lower() == "cuda" and device != "cuda" and log_fn is not None:
            log_fn("WARN: CUDA not available, falling back to CPU.")
        if str(params.device or "").strip().lower() == "mps" and device != "mps" and log_fn is not None:
            log_fn("WARN: MPS not available, falling back to CPU.")

        dt = str(params.delta_type or "none").strip().lower()
        if dt not in {"none", "adapter", "lora", "both"}:
            raise ValueError("delta_type must be none/adapter/lora/both")

        delta_path = None
        if dt != "none":
            delta_path = resolve_best_delta_checkpoint(dt, str(params.delta_checkpoint or "auto"))
            inferred = self._infer_delta_type_from_path(delta_path)
            if inferred is not None and inferred != dt:
                if log_fn is not None:
                    log_fn(
                        f"WARN: delta_type={dt} nhưng checkpoint có vẻ là {inferred} ({os.path.basename(str(delta_path))}). "
                        f"Tự đổi delta_type -> {inferred}."
                    )
                dt = inferred
            if log_fn is not None:
                log_fn(f"Delta: type={dt} ckpt={delta_path}")

        delta_sig = (
            dt,
            delta_path,
            int(params.middle_dim),
            float(params.scaling_factor),
            int(params.rank),
        )

        need_sam = (
            self._predictor is None
            or self._sam_checkpoint != params.sam_checkpoint
            or self._sam_model_type != params.sam_model_type
            or self._delta_sig != delta_sig
            or self._device != device
        )
        if need_sam:
            if log_fn is not None:
                log_fn("Loading SAM checkpoint...")
            sam, _used = load_sam_model(params.sam_checkpoint, params.sam_model_type)
            if dt != "none":
                assert delta_path is not None
                if log_fn is not None:
                    log_fn("Applying delta to SAM...")
                apply_delta_to_sam(
                    sam=sam,
                    delta_type=dt,
                    delta_ckpt_path=str(delta_path),
                    middle_dim=int(params.middle_dim),
                    scaling_factor=float(params.scaling_factor),
                    rank=int(params.rank),
                )
            sam.to(device=device)
            self._predictor = SamPredictor(sam)
            self._sam_checkpoint = params.sam_checkpoint
            self._sam_model_type = params.sam_model_type
            self._delta_sig = delta_sig
            self._device = device
            if log_fn is not None:
                log_fn(f"SAM ready (type={params.sam_model_type}, device={device}).")

        ckpt_path = str(params.gdino_checkpoint or "").strip()
        if not ckpt_path:
            raise FileNotFoundError("GroundingDINO checkpoint path is required.")

        ckpt_lower = ckpt_path.lower()
        ckpt_is_dir = os.path.isdir(ckpt_path)
        ckpt_is_file = os.path.isfile(ckpt_path)
        ckpt_is_explicit_file = ckpt_lower.endswith((".pth", ".pt", ".safetensors", ".bin"))
        use_hf_id = (not ckpt_is_dir) and (not ckpt_is_file) and (not ckpt_is_explicit_file)

        if use_hf_id or ckpt_is_dir:
            cfg_id = ckpt_path
        else:
            cfg_id = self._resolve_gdino_config_id(params)

        need_gdino = (
            self._gdino is None
            or self._processor is None
            or self._gdino_checkpoint != ckpt_path
            or self._gdino_config_id != cfg_id
            or self._device != device
        )
        if need_gdino:
            if log_fn is not None:
                if use_hf_id:
                    log_fn("Loading GroundingDINO from HuggingFace...")
                else:
                    log_fn("Loading GroundingDINO (config from HuggingFace, weights from .pth)...")

            if ckpt_is_dir:
                processor = AutoProcessor.from_pretrained(ckpt_path)
                gdino = GroundingDinoForObjectDetection.from_pretrained(ckpt_path)
            elif use_hf_id:
                processor = AutoProcessor.from_pretrained(ckpt_path)
                gdino = GroundingDinoForObjectDetection.from_pretrained(ckpt_path)
            else:
                if not ckpt_is_file:
                    raise FileNotFoundError(f"GroundingDINO checkpoint not found: {ckpt_path}")
                state_dict = self._load_gdino_state_dict(ckpt_path)
                processor = AutoProcessor.from_pretrained(cfg_id)
                config = GroundingDinoConfig.from_pretrained(cfg_id)
                gdino = GroundingDinoForObjectDetection(config)
                missing, unexpected = gdino.load_state_dict(state_dict, strict=False)
                if log_fn is not None and (missing or unexpected):
                    log_fn(f"WARN: GroundingDINO missing={len(missing)} unexpected={len(unexpected)} keys.")
            gdino.to(device)
            gdino.eval()
            self._processor = processor
            self._gdino = gdino
            self._gdino_checkpoint = ckpt_path
            self._gdino_config_id = cfg_id
            if log_fn is not None:
                log_fn(f"GroundingDINO ready (ckpt={ckpt_path}, config={cfg_id}).")

        return self._predictor, self._processor, self._gdino, device

    def run(self, image_path: str, params: SamDinoParams, *, stop_checker=None, log_fn=None) -> dict:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.path.isfile(params.sam_checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found: {params.sam_checkpoint}")

        os.makedirs(params.output_dir, exist_ok=True)

        self._ensure_import_paths()
        from sam_dino.pipeline import StopRequested, process_one_image_text_box_sam_fullimage, safe_basename

        predictor, processor, gdino, device = self._ensure_models(params, log_fn=log_fn)

        base = safe_basename(image_path)
        if log_fn is not None:
            log_fn("Running detection + SAM masks...")
        n_dets, n_masks, overlay_path, mask_path = process_one_image_text_box_sam_fullimage(
            image_path=image_path,
            out_dir=params.output_dir,
            predictor=predictor,
            processor=processor,
            gdino=gdino,
            device=device,
            text_queries=list(params.text_queries),
            box_threshold=float(params.box_threshold),
            text_threshold=float(params.text_threshold),
            max_dets=int(params.max_dets),
            overlay_alpha=float(params.overlay_alpha),
            seed=int(params.seed),
            invert_mask=bool(params.invert_mask),
            sam_min_component_area=int(params.sam_min_component_area),
            sam_dilate_iters=int(params.sam_dilate_iters),
            stop_checker=stop_checker,
        )
        if log_fn is not None:
            log_fn(f"Done. dets={n_dets}, masks_saved={n_masks}")

        return {
            "mask_path": mask_path,
            "overlay_path": overlay_path,
            "output_dir": params.output_dir,
            "dets": int(n_dets),
            "masks_saved": int(n_masks),
        }

    def run_isolate(
        self,
        image_path: str,
        params: SamDinoParams,
        *,
        target_labels: Sequence[str],
        outside_value: int = 0,
        crop_to_bbox: bool = False,
        stop_checker=None,
        log_fn=None,
    ) -> dict:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.path.isfile(params.sam_checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found: {params.sam_checkpoint}")

        os.makedirs(params.output_dir, exist_ok=True)

        self._ensure_import_paths()
        from sam_dino.pipeline import StopRequested, process_one_image_text_box_sam_isolate

        predictor, processor, gdino, device = self._ensure_models(params, log_fn=log_fn)

        if log_fn is not None:
            log_fn("Running isolate (GroundingDINO -> SAM union)...")

        n_dets, masks_saved, overlay_path, mask_path, isolate_path = process_one_image_text_box_sam_isolate(
            image_path=image_path,
            out_dir=params.output_dir,
            predictor=predictor,
            processor=processor,
            gdino=gdino,
            device=device,
            text_queries=list(params.text_queries),
            target_labels=list(target_labels),
            box_threshold=float(params.box_threshold),
            text_threshold=float(params.text_threshold),
            max_dets=int(params.max_dets),
            outside_value=int(outside_value),
            crop_to_bbox=bool(crop_to_bbox),
            overlay_alpha=float(params.overlay_alpha),
            seed=int(params.seed),
            invert_mask=bool(params.invert_mask),
            sam_min_component_area=int(params.sam_min_component_area),
            sam_dilate_iters=int(params.sam_dilate_iters),
            stop_checker=stop_checker,
        )

        return {
            "mask_path": mask_path,
            "overlay_path": overlay_path,
            "isolate_path": isolate_path,
            "output_dir": params.output_dir,
            "dets": int(n_dets),
            "masks_saved": int(masks_saved),
        }
