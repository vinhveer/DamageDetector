from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from torch_runtime import describe_device_fallback, get_torch, select_device_str


def default_dinov2_checkpoint() -> str:
    base_dir = Path(__file__).resolve().parent / "models"
    preferred = base_dir / "surface_crack_image_detection"
    if preferred.is_dir():
        return str(preferred.resolve())
    return str((base_dir / "dinov2-small-imagenet1k-1-layer").resolve())


class DinoV2ClassifierRunner:
    def __init__(self) -> None:
        self._checkpoint: str | None = None
        self._device: str | None = None
        self._processor: Any | None = None
        self._model: Any | None = None

    def ensure_model_loaded(self, checkpoint_path: str, *, device_preference: str = "auto", log_fn=None) -> tuple[Any, Any, str]:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        device = select_device_str(device_preference)
        fallback = describe_device_fallback(device_preference, device)
        if fallback is not None and log_fn is not None:
            log_fn(fallback)

        resolved = str(checkpoint_path or "").strip()
        if not resolved:
            raise FileNotFoundError("DINOv2 classifier checkpoint path is required.")

        needs_reload = (
            self._processor is None
            or self._model is None
            or self._checkpoint != resolved
            or self._device != device
        )
        if not needs_reload:
            return self._processor, self._model, device

        local_files_only = Path(resolved).exists()
        if log_fn is not None:
            source = "local folder" if local_files_only else "cached model id"
            log_fn(f"Loading DINOv2 classifier from {source}...")
        try:
            processor = AutoImageProcessor.from_pretrained(resolved, local_files_only=local_files_only)
            model = AutoModelForImageClassification.from_pretrained(resolved, local_files_only=local_files_only)
        except Exception as exc:
            raise RuntimeError(
                "Cannot load DINOv2 classifier locally.\n\n"
                "Fix options:\n"
                "1) Point dinov2_checkpoint to a local HuggingFace classifier folder.\n"
                "2) Or pre-download the DINOv2 image-classification repo into dino/models.\n\n"
                f"dinov2_checkpoint={resolved}"
            ) from exc

        model.to(device)
        model.eval()
        self._checkpoint = resolved
        self._device = device
        self._processor = processor
        self._model = model
        return processor, model, device

    def classify_crops(
        self,
        *,
        checkpoint_path: str,
        images: Sequence[Any],
        device_preference: str = "auto",
        batch_size: int = 8,
        top_k: int = 3,
        log_fn=None,
    ) -> list[dict[str, Any]]:
        torch = get_torch()

        if not images:
            return []

        processor, model, device = self.ensure_model_loaded(
            checkpoint_path,
            device_preference=device_preference,
            log_fn=log_fn,
        )
        effective_batch_size = max(1, int(batch_size))
        effective_top_k = max(1, int(top_k))
        id2label = dict(getattr(getattr(model, "config", None), "id2label", {}) or {})
        outputs_payload: list[dict[str, Any]] = []
        for start in range(0, len(images), effective_batch_size):
            batch = list(images[start : start + effective_batch_size])
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {name: value.to(device) if hasattr(value, "to") else value for name, value in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
            top_probs, top_ids = torch.topk(probs, k=min(effective_top_k, probs.shape[-1]), dim=-1)
            for row_probs, row_ids in zip(top_probs.detach().cpu().tolist(), top_ids.detach().cpu().tolist()):
                predictions = []
                for prob, class_id in zip(row_probs, row_ids):
                    label = id2label.get(int(class_id), str(class_id))
                    predictions.append(
                        {
                            "label": str(label),
                            "confidence": float(prob),
                            "class_id": int(class_id),
                        }
                    )
                top_prediction = predictions[0] if predictions else {"label": "", "confidence": 0.0, "class_id": -1}
                outputs_payload.append(
                    {
                        "label": str(top_prediction.get("label") or ""),
                        "confidence": float(top_prediction.get("confidence") or 0.0),
                        "class_id": int(top_prediction.get("class_id") or -1),
                        "top_predictions": predictions,
                    }
                )
        return outputs_payload
