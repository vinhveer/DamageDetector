from __future__ import annotations

import os
import re
import tempfile
import json
from importlib import import_module

from sam.runtime import infer_sam_model_type_from_state_dict, load_checkpoint_state_dict

from torch_runtime import get_torch


def infer_delta_type_from_path(path: str | None) -> str | None:
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


def resolve_best_delta_checkpoint(delta_type: str, delta_checkpoint: str) -> str | None:
    if not delta_type or delta_type.lower() == "none":
        return None
    if delta_checkpoint and delta_checkpoint.lower() != "auto":
        if not os.path.isfile(delta_checkpoint):
            raise FileNotFoundError(f"Delta checkpoint not found: {delta_checkpoint}")
        return delta_checkpoint

    normalized = delta_type.lower().strip()
    candidates: list[str] = []
    search_dirs = [os.getcwd(), os.path.join(os.getcwd(), "checkpoints")]
    patterns = {
        "adapter": [r"(?i)adapter.*\.pth$", r"(?i)delta.*adapter.*\.pth$"],
        "lora": [r"(?i)lora.*\.pth$", r"(?i)delta.*lora.*\.pth$"],
        "both": [r"(?i)adapter.*lora.*\.pth$", r"(?i)lora.*adapter.*\.pth$"],
    }
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(path)[1].lower() != ".pth":
                continue
            if re.search(r"(?i)^sam_.*\.pth$", name):
                continue
            if any(re.search(pattern, name) for pattern in patterns.get(normalized, [])):
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"Cannot auto-find delta checkpoint for delta_type={delta_type!r}.")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _sidecar_dir(delta_checkpoint: str | None) -> str | None:
    if not delta_checkpoint:
        return None
    directory = os.path.dirname(os.path.abspath(str(delta_checkpoint)))
    return directory if os.path.isdir(directory) else None


def load_inference_config(delta_checkpoint: str | None) -> dict:
    directory = _sidecar_dir(delta_checkpoint)
    if directory is None:
        return {}
    path = os.path.join(directory, "inference_config.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def infer_decoder_type_from_path(delta_checkpoint: str | None) -> str | None:
    if not delta_checkpoint or not os.path.isfile(delta_checkpoint):
        return None
    config = load_inference_config(delta_checkpoint)
    decoder_type = str(config.get("decoder_type", "")).strip().lower()
    if decoder_type in {"baseline", "hq"}:
        return decoder_type
    torch = get_torch()
    try:
        state_dict = torch.load(delta_checkpoint, map_location="cpu", weights_only=True)
    except Exception:
        state_dict = torch.load(delta_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        return None
    if any(str(key).startswith("mask_decoder.hf_") for key in state_dict.keys()):
        return "hq"
    return "baseline"


def resolve_decoder_type(delta_checkpoint: str | None, decoder_type: str | None = None) -> str:
    requested = str(decoder_type or "auto").strip().lower()
    if requested in {"baseline", "hq"}:
        return requested
    inferred = infer_decoder_type_from_path(delta_checkpoint)
    return inferred or "baseline"


def resolve_centerline_head(delta_checkpoint: str | None, centerline_head: bool | None = None) -> bool:
    if centerline_head is not None:
        return bool(centerline_head)
    config = load_inference_config(delta_checkpoint)
    return bool(config.get("centerline_head", False))


def resolve_predict_threshold(delta_checkpoint: str | None, threshold: str | float | None) -> float:
    if threshold is not None and str(threshold).strip().lower() != "auto":
        return float(threshold)

    directory = _sidecar_dir(delta_checkpoint)
    if directory is not None:
        best_threshold_path = os.path.join(directory, "best_threshold.txt")
        if os.path.isfile(best_threshold_path):
            try:
                with open(best_threshold_path, "r", encoding="utf-8") as f:
                    return float(f.read().strip())
            except Exception:
                pass

    config = load_inference_config(delta_checkpoint)
    best_threshold = config.get("best_threshold")
    if best_threshold is not None:
        try:
            return float(best_threshold)
        except Exception:
            pass
    return 0.5


def resolve_predict_mode(delta_checkpoint: str | None, predict_mode: str | None) -> str:
    if predict_mode is not None:
        normalized = str(predict_mode).strip().lower()
        if normalized and normalized != "auto":
            return normalized
    config = load_inference_config(delta_checkpoint)
    mode = str(config.get("predict_mode", "tile_full_box")).strip().lower()
    return mode or "tile_full_box"


def resolve_image_size(delta_checkpoint: str | None, image_size: int | None = None) -> int:
    size = int(image_size) if image_size is not None else -1
    if size > 0:
        return size
    config = load_inference_config(delta_checkpoint)
    try:
        size = int(config.get("img_size", 512))
    except Exception:
        size = 512
    return max(1, int(size))


def resolve_tile_settings(delta_checkpoint: str | None, tile_size: int | None, tile_overlap: int | None) -> tuple[int, int]:
    config = load_inference_config(delta_checkpoint)
    size = int(tile_size) if tile_size is not None else -1
    overlap = int(tile_overlap) if tile_overlap is not None else -1
    if size <= 0:
        try:
            size = int(config.get("img_size", 512))
        except Exception:
            size = 512
    if overlap < 0:
        try:
            overlap = int(config.get("tile_overlap", size // 2))
        except Exception:
            overlap = size // 2
    overlap = max(0, min(overlap, max(0, size - 1)))
    return int(size), int(overlap)


def resolve_refine_settings(
    delta_checkpoint: str | None,
    *,
    refine_tile_size: int | None = None,
    refine_tile_sizes: list[int] | tuple[int, ...] | None = None,
    refine_max_rois: int | None = None,
    refine_roi_padding: int | None = None,
    refine_merge_mode: str | None = None,
    refine_score_threshold: float | None = None,
    positive_band_low: float | None = None,
    positive_band_high: float | None = None,
) -> dict:
    config = load_inference_config(delta_checkpoint)
    tile_size = int(refine_tile_size) if refine_tile_size is not None else -1
    if tile_size <= 0:
        try:
            tile_size = int(config.get("refine_tile_size", 768))
        except Exception:
            tile_size = 768
    tile_sizes = refine_tile_sizes
    if tile_sizes is None:
        raw_sizes = config.get("refine_tile_sizes")
        if isinstance(raw_sizes, (list, tuple)):
            tile_sizes = [int(v) for v in raw_sizes if int(v) > 0]
        else:
            tile_sizes = []
    else:
        tile_sizes = [int(v) for v in tile_sizes if int(v) > 0]
    if not tile_sizes:
        tile_sizes = [max(1, int(tile_size))]

    max_rois = int(refine_max_rois) if refine_max_rois is not None else -1
    if max_rois <= 0:
        try:
            max_rois = int(config.get("refine_max_rois", 16))
        except Exception:
            max_rois = 16

    roi_padding = int(refine_roi_padding) if refine_roi_padding is not None else -1
    if roi_padding < 0:
        try:
            roi_padding = int(config.get("refine_roi_padding", 64))
        except Exception:
            roi_padding = 64

    merge_mode = str(refine_merge_mode or config.get("refine_merge_mode", "weighted_replace")).strip().lower()
    if not merge_mode:
        merge_mode = "weighted_replace"

    if refine_score_threshold is None:
        try:
            score_threshold = float(config.get("refine_score_threshold", 0.15))
        except Exception:
            score_threshold = 0.15
    else:
        score_threshold = float(refine_score_threshold)

    if positive_band_low is None:
        try:
            band_low = float(config.get("refine_positive_band_low", 0.20))
        except Exception:
            band_low = 0.20
    else:
        band_low = float(positive_band_low)

    if positive_band_high is None:
        try:
            band_high = float(config.get("refine_positive_band_high", 0.90))
        except Exception:
            band_high = 0.90
    else:
        band_high = float(positive_band_high)

    return {
        "refine_tile_size": max(1, int(tile_size)),
        "refine_tile_sizes": tile_sizes,
        "refine_max_rois": max(1, int(max_rois)),
        "refine_roi_padding": max(0, int(roi_padding)),
        "refine_merge_mode": merge_mode,
        "refine_score_threshold": float(score_threshold),
        "positive_band_low": float(band_low),
        "positive_band_high": float(band_high),
    }


def load_sam_model(
    checkpoint_path: str,
    requested_model_type: str,
    *,
    image_size: int = 1024,
    pixel_mean=None,
    pixel_std=None,
    decoder_type: str = "baseline",
    centerline_head: bool = False,
):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")
    state_dict = load_checkpoint_state_dict(checkpoint_path)
    inferred = infer_sam_model_type_from_state_dict(state_dict)
    requested = (requested_model_type or "auto").strip().lower()
    if requested == "auto":
        if inferred is None:
            raise RuntimeError(
                "Cannot infer SAM model type from checkpoint. Choose the correct one explicitly: vit_b, vit_l, or vit_h."
            )
        model_type = inferred
    else:
        model_type = inferred or requested
    from sam_finetune.segment_anything import sam_model_registry

    if model_type not in sam_model_registry:
        raise ValueError(f"Unknown SAM model type: {model_type!r}")
    kwargs = {
        "image_size": int(image_size),
        "num_classes": 1,
        "checkpoint": None,
        "decoder_type": resolve_decoder_type(None, decoder_type),
        "centerline_head": bool(centerline_head),
    }
    if pixel_mean is not None:
        kwargs["pixel_mean"] = pixel_mean
    if pixel_std is not None:
        kwargs["pixel_std"] = pixel_std
    sam_model, _ = sam_model_registry[model_type](**kwargs)
    try:
        sam_model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"SAM checkpoint/model type mismatch. requested={requested_model_type!r}, inferred={inferred!r}.\n{exc}"
        ) from exc
    return sam_model, model_type


def apply_delta_to_sam(
    *,
    sam,
    delta_type: str,
    delta_ckpt_path: str,
    middle_dim: int,
    scaling_factor: float,
    rank: int,
) -> None:
    torch = get_torch()
    try:
        delta_state = torch.load(delta_ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        delta_state = torch.load(delta_ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(delta_state, dict):
        raise TypeError(f"Unsupported delta checkpoint format: {type(delta_state)} ({delta_ckpt_path})")

    def _infer_delta_type_from_state(state: dict) -> str:
        keys = [str(k) for k in state.keys()]
        has_lora_suffix = any(k.startswith("w_a_") and k.endswith("_l") for k in keys) or any(
            k.startswith("w_b_") and k.endswith("_l") for k in keys
        )
        if has_lora_suffix:
            return "both"
        has_adapter_markers = any(k.startswith(("w_c_", "w_d_")) for k in keys) or any(k.endswith("_bia") for k in keys)
        if has_adapter_markers:
            return "adapter"
        has_lora_markers = any(k.startswith("w_a_") for k in keys) and any(k.startswith("w_b_") for k in keys)
        if has_lora_markers:
            return "lora"
        return "none"

    normalized = delta_type.lower().strip()
    inferred = _infer_delta_type_from_state(delta_state)
    if inferred in {"adapter", "lora", "both"} and inferred != normalized:
        print(f"WARN: delta_type={normalized} but checkpoint looks like {inferred}. Auto-using {inferred}.", flush=True)
        normalized = inferred

    if normalized == "adapter":
        ckpt_middle_dim = None
        for idx in range(1000):
            key = f"w_a_{idx:03d}"
            value = delta_state.get(key, None)
            if isinstance(value, torch.Tensor):
                ckpt_middle_dim = int(value.shape[0])
                break
        if ckpt_middle_dim is not None and int(middle_dim) != ckpt_middle_dim:
            middle_dim = ckpt_middle_dim
        wrapper_module = import_module("sam_finetune.delta.sam_adapter_image_encoder")
        wrapper = wrapper_module.Adapter_Sam(sam, int(middle_dim), float(scaling_factor))
    elif normalized == "lora":
        wrapper_module = import_module("sam_finetune.delta.sam_lora_image_encoder")
        wrapper = wrapper_module.LoRA_Sam(sam, int(rank))
    elif normalized == "both":
        ckpt_middle_dim = None
        for idx in range(1000):
            key = f"w_a_{idx:03d}"
            value = delta_state.get(key, None)
            if isinstance(value, torch.Tensor):
                ckpt_middle_dim = int(value.shape[0])
                break
        if ckpt_middle_dim is not None and int(middle_dim) != ckpt_middle_dim:
            middle_dim = ckpt_middle_dim
        wrapper_module = import_module("sam_finetune.delta.sam_adapter_lora_image_encoder")
        wrapper = wrapper_module.LoRA_Adapter_Sam(sam, int(middle_dim), int(rank))
    else:
        raise ValueError(f"Unknown delta_type: {delta_type!r}")

    if normalized in {"adapter", "both"}:
        tensor_specs = [
            ("w_down_attn", "w_a_{i:03d}", "w_a_{i:03d}_bia"),
            ("w_up_attn", "w_b_{i:03d}", "w_b_{i:03d}_bia"),
            ("w_down_mlp", "w_c_{i:03d}", "w_c_{i:03d}_bia"),
            ("w_up_mlp", "w_d_{i:03d}", "w_d_{i:03d}_bia"),
        ]
        for attr, w_tpl, b_tpl in tensor_specs:
            layers = getattr(wrapper, attr, None)
            if not layers:
                continue
            for index, layer in enumerate(layers):
                w_key = w_tpl.format(i=index)
                if w_key not in delta_state:
                    weight = getattr(layer, "weight", None)
                    if isinstance(weight, torch.Tensor):
                        delta_state[w_key] = torch.zeros_like(weight, device="cpu")
                b_key = b_tpl.format(i=index)
                if b_key not in delta_state:
                    bias = getattr(layer, "bias", None)
                    if isinstance(bias, torch.Tensor):
                        delta_state[b_key] = torch.zeros_like(bias, device="cpu")

    sam_state = sam.state_dict()
    for key, value in sam_state.items():
        if ("prompt_encoder" not in key) and ("mask_decoder" not in key):
            continue
        if key not in delta_state:
            delta_state[key] = value
            continue
        delta_value = delta_state[key]
        if isinstance(delta_value, torch.Tensor) and isinstance(value, torch.Tensor) and tuple(delta_value.shape) != tuple(value.shape):
            delta_state[key] = value

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as handle:
        tmp_path = handle.name
    try:
        torch.save(delta_state, tmp_path)
        wrapper.load_delta_parameters(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
