from __future__ import annotations

import os
from typing import Any

from inference_api.contracts import InferenceRequest


TITLE_BY_MODE: dict[str, str] = {
    "sam_dino": "Predict SAM + DINO",
    "sam_dino_ft": "Predict SAM + DINO + Finetune",
    "sam_only": "Predict SAM Only",
    "sam_only_ft": "Predict SAM Only + Finetune",
    "sam_tiled": "Predict SAM + DINO Tiled",
    "unet_only": "Predict UNet Only",
    "unet_dino": "Predict UNet + DINO",
    "isolate": "Isolate Object",
}

PAGES_BY_MODE: dict[str, list[str]] = {
    "sam_dino": ["SAM", "DINO"],
    "sam_dino_ft": ["SAM", "DINO"],
    "sam_only": ["SAM"],
    "sam_only_ft": ["SAM"],
    "sam_tiled": ["SAM", "DINO"],
    "unet_only": ["UNet"],
    "unet_dino": ["UNet", "DINO"],
    "isolate": ["SAM", "DINO"],
}


def normalize_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized == "unet":
        return "unet_dino"
    return normalized


def prediction_title(mode: str) -> str:
    normalized = normalize_mode(mode)
    return TITLE_BY_MODE.get(normalized, f"Predict {normalized}")


def settings_pages_for_mode(mode: str) -> list[str]:
    normalized = normalize_mode(mode)
    return list(PAGES_BY_MODE.get(normalized, ["SAM", "DINO", "UNet"]))


def missing_editor_settings(mode: str, settings: dict[str, Any]) -> tuple[str, str] | None:
    normalized = normalize_mode(mode)
    if normalized in {"sam_dino", "sam_dino_ft", "sam_tiled", "unet_dino", "isolate"}:
        missing = _require_dino_settings(settings)
        if missing is not None:
            return missing
    if normalized in {"sam_dino", "sam_dino_ft", "sam_only", "sam_only_ft", "sam_tiled", "isolate"}:
        missing = _require_sam_settings(settings)
        if missing is not None:
            return missing
    if normalized in {"sam_dino", "sam_dino_ft"}:
        if not _split_queries(settings.get("text_queries")):
            return ("DINO", "Text queries are required.")
    if normalized in {"sam_dino_ft", "sam_only_ft"}:
        missing = _require_delta_settings(settings)
        if missing is not None:
            return missing
    if normalized in {"unet_only", "unet_dino"}:
        missing = _require_unet_settings(settings)
        if missing is not None:
            return missing
    if normalized not in TITLE_BY_MODE:
        return ("SAM", f"Unknown predict mode: {mode}")
    return None


def build_editor_request(
    mode: str,
    settings: dict[str, Any],
    *,
    image_path: str | None = None,
    image_paths: list[str] | None = None,
    roi_box: tuple[int, int, int, int] | None = None,
    output_dir: str | None = None,
    target_labels: list[str] | None = None,
    outside_value: int | None = None,
    crop_to_bbox: bool | None = None,
    max_depth: int | None = None,
    min_box_px: int | None = None,
) -> InferenceRequest:
    normalized = normalize_mode(mode)
    missing = missing_editor_settings(normalized, settings)
    if missing is not None:
        _, message = missing
        raise ValueError(message)
    out_dir = str(output_dir or "")
    if normalized == "sam_dino":
        return InferenceRequest(
            workflow="sam_dino",
            image_path=image_path,
            image_paths=list(image_paths) if image_paths is not None else None,
            roi_box=roi_box,
            params={
                "sam": _build_sam_params(settings, output_dir=out_dir, roi_box=roi_box, require_dino=False, use_delta=False),
                "dino": _build_dino_params(settings, output_dir=out_dir, roi_box=roi_box),
            },
            client_tag="ground_truth_editor",
            source="editor",
        )
    if normalized == "sam_dino_ft":
        return InferenceRequest(
            workflow="sam_dino_ft",
            image_path=image_path,
            image_paths=list(image_paths) if image_paths is not None else None,
            roi_box=roi_box,
            params={
                "sam": _build_sam_params(settings, output_dir=out_dir, roi_box=roi_box, require_dino=False, use_delta=True),
                "dino": _build_dino_params(settings, output_dir=out_dir, roi_box=roi_box),
            },
            client_tag="ground_truth_editor",
            source="editor",
        )
    if normalized == "sam_only":
        return InferenceRequest(
            workflow="sam_only",
            image_path=image_path,
            image_paths=list(image_paths) if image_paths is not None else None,
            roi_box=roi_box,
            params={"sam": _build_sam_params(settings, output_dir=out_dir, roi_box=roi_box, require_dino=False, use_delta=False)},
            client_tag="ground_truth_editor",
            source="editor",
        )
    if normalized == "sam_only_ft":
        return InferenceRequest(
            workflow="sam_only_ft",
            image_path=image_path,
            image_paths=list(image_paths) if image_paths is not None else None,
            roi_box=roi_box,
            params={"sam": _build_sam_params(settings, output_dir=out_dir, roi_box=roi_box, require_dino=False, use_delta=True)},
            client_tag="ground_truth_editor",
            source="editor",
        )
    if normalized == "sam_tiled":
        return InferenceRequest(
            workflow="sam_tiled",
            image_path=image_path,
            image_paths=list(image_paths) if image_paths is not None else None,
            roi_box=roi_box,
            params={
                "sam": _build_sam_params(settings, output_dir=out_dir, roi_box=roi_box, require_dino=False, use_delta=False),
                "dino": _build_dino_params(settings, output_dir=out_dir, roi_box=roi_box),
                "target_labels": list(target_labels or ["crack"]),
                "max_depth": int(max_depth or 3),
                "min_box_px": int(min_box_px or 48),
            },
            client_tag="ground_truth_editor",
            source="editor",
        )
    if normalized == "unet_only":
        return InferenceRequest(
            workflow="unet_only",
            image_path=image_path,
            image_paths=list(image_paths) if image_paths is not None else None,
            roi_box=roi_box,
            params={"unet": _build_unet_params(settings, output_dir=out_dir, roi_box=roi_box)},
            client_tag="ground_truth_editor",
            source="editor",
        )
    if normalized == "unet_dino":
        return InferenceRequest(
            workflow="unet_dino",
            image_path=image_path,
            image_paths=list(image_paths) if image_paths is not None else None,
            roi_box=roi_box,
            params={
                "unet": _build_unet_params(settings, output_dir=out_dir, roi_box=roi_box),
                "dino": _build_dino_params(settings, output_dir=out_dir, roi_box=roi_box),
            },
            client_tag="ground_truth_editor",
            source="editor",
        )
    if normalized == "isolate":
        labels = [label.strip() for label in (target_labels or []) if str(label).strip()]
        return InferenceRequest(
            workflow="isolate",
            image_path=image_path,
            roi_box=roi_box,
            params={
                "sam": _build_sam_params(settings, output_dir=out_dir, roi_box=roi_box, require_dino=False, use_delta=False),
                "dino": _build_dino_params(settings, output_dir=out_dir, roi_box=roi_box),
                "target_labels": labels,
                "outside_value": int(outside_value or 0),
                "crop_to_bbox": bool(crop_to_bbox or False),
            },
            client_tag="ground_truth_editor",
            source="editor",
        )
    raise ValueError(f"Unknown predict mode: {mode}")


def _require_sam_settings(settings: dict[str, Any]) -> tuple[str, str] | None:
    sam_ckpt = str(settings.get("sam_checkpoint") or "").strip()
    if not sam_ckpt or not os.path.isfile(sam_ckpt):
        return ("SAM", "SAM checkpoint is required (file not found).")
    return None


def _require_dino_settings(settings: dict[str, Any]) -> tuple[str, str] | None:
    gdino_ckpt = str(settings.get("dino_checkpoint") or "").strip()
    if not gdino_ckpt:
        return ("DINO", "GroundingDINO checkpoint is required.")
    lower = gdino_ckpt.lower()
    if lower.endswith((".pth", ".pt", ".safetensors", ".bin")) and not os.path.exists(gdino_ckpt):
        return ("DINO", f"GroundingDINO checkpoint not found: {gdino_ckpt}")
    config_id = str(settings.get("dino_config_id") or "").strip()
    if not config_id:
        return ("DINO", "GroundingDINO config is required.")
    return None


def _require_delta_settings(settings: dict[str, Any]) -> tuple[str, str] | None:
    delta_ckpt = str(settings.get("delta_checkpoint") or "").strip()
    if not delta_ckpt:
        return ("SAM", "Delta checkpoint is required (set to 'auto' or choose a file).")
    lower = delta_ckpt.lower()
    if lower != "auto" and lower.endswith((".pth", ".pt", ".safetensors", ".bin")) and not os.path.exists(delta_ckpt):
        return ("SAM", f"Delta checkpoint not found: {delta_ckpt}")
    delta_type = str(settings.get("delta_type") or "").strip().lower()
    if delta_type not in {"adapter", "lora", "both"}:
        return ("SAM", "Delta type must be adapter, lora, or both.")
    return None


def _require_unet_settings(settings: dict[str, Any]) -> tuple[str, str] | None:
    model_path = str(settings.get("unet_model") or "").strip()
    if not model_path or not os.path.isfile(model_path):
        return ("UNet", "UNet model is required (file not found).")
    return None


def _split_queries(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _build_sam_params(
    settings: dict[str, Any],
    *,
    output_dir: str,
    roi_box: tuple[int, int, int, int] | None,
    require_dino: bool,
    use_delta: bool,
) -> dict[str, Any]:
    params = {
        "sam_checkpoint": str(settings.get("sam_checkpoint") or "").strip(),
        "sam_model_type": str(settings.get("sam_model_type") or "auto"),
        "invert_mask": bool(settings.get("invert_mask") or False),
        "sam_min_component_area": int(settings.get("min_area") or 0),
        "sam_dilate_iters": int(settings.get("dilate") or 0),
        "device": str(settings.get("device") or "auto"),
        "output_dir": output_dir or "results_sam_dino",
    }
    if use_delta:
        params["delta_type"] = str(settings.get("delta_type") or "")
        params["delta_checkpoint"] = str(settings.get("delta_checkpoint") or "").strip()
        params["middle_dim"] = int(settings.get("middle_dim") or 32)
        params["scaling_factor"] = float(settings.get("scaling_factor") or 0.2)
        params["rank"] = int(settings.get("rank") or 4)
    if require_dino:
        params["gdino_checkpoint"] = str(settings.get("dino_checkpoint") or "").strip()
        params["gdino_config_id"] = str(settings.get("dino_config_id") or "").strip()
        params["text_queries"] = _split_queries(settings.get("text_queries"))
        params["box_threshold"] = float(settings.get("box_threshold") or 0.25)
        params["text_threshold"] = float(settings.get("text_threshold") or 0.25)
        params["max_dets"] = int(settings.get("max_dets") or 20)
    if roi_box is not None:
        params["roi_box"] = tuple(int(x) for x in roi_box)
    return params


def _build_unet_params(
    settings: dict[str, Any],
    *,
    output_dir: str,
    roi_box: tuple[int, int, int, int] | None,
) -> dict[str, Any]:
    params = {
        "model_path": str(settings.get("unet_model") or "").strip(),
        "output_dir": output_dir or "results_unet",
        "threshold": float(settings.get("unet_threshold") or 0.5),
        "apply_postprocessing": bool(settings.get("unet_post") or False),
        "mode": str(settings.get("unet_mode") or "tile"),
        "input_size": int(settings.get("unet_input_size") or 256),
        "tile_overlap": int(settings.get("unet_overlap") or 0),
        "tile_batch_size": int(settings.get("unet_tile_batch") or 4),
        "device": str(settings.get("device") or "auto"),
    }
    if roi_box is not None:
        params["roi_box"] = tuple(int(x) for x in roi_box)
    return params


def _build_dino_params(
    settings: dict[str, Any],
    *,
    output_dir: str,
    roi_box: tuple[int, int, int, int] | None,
) -> dict[str, Any]:
    params = {
        "gdino_checkpoint": str(settings.get("dino_checkpoint") or "").strip(),
        "gdino_config_id": str(settings.get("dino_config_id") or "").strip(),
        "text_queries": _split_queries(settings.get("text_queries")),
        "box_threshold": float(settings.get("box_threshold") or 0.25),
        "text_threshold": float(settings.get("text_threshold") or 0.25),
        "max_dets": int(settings.get("max_dets") or 20),
        "device": str(settings.get("device") or "auto"),
        "output_dir": output_dir or "results_sam_dino",
    }
    if roi_box is not None:
        params["roi_box"] = tuple(int(x) for x in roi_box)
    return params
