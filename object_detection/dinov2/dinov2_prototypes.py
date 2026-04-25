from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from torch_runtime import describe_device_fallback, get_torch, select_device_str

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _model_roots() -> list[Path]:
    module_dir = Path(__file__).resolve().parent
    return [module_dir / "models", module_dir.parent / "dino" / "models"]


def default_dinov2_embedding_checkpoint() -> str:
    for base_dir in _model_roots():
        preferred = base_dir / "dinov2-small"
        if preferred.is_dir():
            return str(preferred.resolve())
    for base_dir in _model_roots():
        fallback = base_dir / "dinov2-small-imagenet1k-1-layer"
        if fallback.is_dir():
            return str(fallback.resolve())
    return str((_model_roots()[0] / "dinov2-small-imagenet1k-1-layer").resolve())


def _normalize_label(text: str) -> str:
    return " ".join(str(text or "").replace("_", " ").replace("-", " ").strip().lower().split())


def _label_allowed(label: str, allowed: set[str]) -> bool:
    normalized = _normalize_label(label)
    if not allowed:
        return True
    for token in allowed:
        if token == normalized or token in normalized or normalized in token:
            return True
    return False


class DinoV2PrototypeRunner:
    def __init__(self) -> None:
        self._checkpoint: str | None = None
        self._device: str | None = None
        self._processor: Any | None = None
        self._model: Any | None = None
        self._prototype_cache: dict[tuple[str, str, str, tuple[str, ...]], dict[str, Any]] = {}

    def ensure_model_loaded(self, checkpoint_path: str, *, device_preference: str = "auto", log_fn=None) -> tuple[Any, Any, str]:
        from transformers import AutoImageProcessor, AutoModel

        device = select_device_str(device_preference)
        fallback = describe_device_fallback(device_preference, device)
        if fallback is not None and log_fn is not None:
            log_fn(fallback)

        resolved = str(checkpoint_path or "").strip()
        if not resolved:
            raise FileNotFoundError("DINOv2 embedding checkpoint path is required.")

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
            log_fn(f"Loading DINOv2 embedding backbone from {source}...")
        try:
            processor = AutoImageProcessor.from_pretrained(resolved, local_files_only=local_files_only)
            model = AutoModel.from_pretrained(resolved, local_files_only=local_files_only)
        except Exception as exc:
            raise RuntimeError(
                "Cannot load DINOv2 embedding model locally.\n\n"
                "Fix options:\n"
                "1) Point dinov2_checkpoint to a local HuggingFace DINOv2 backbone folder.\n"
                "2) Or pre-download the DINOv2 backbone repo into dinov2/models (or legacy dino/models).\n\n"
                f"dinov2_checkpoint={resolved}"
            ) from exc

        model.to(device)
        model.eval()
        self._checkpoint = resolved
        self._device = device
        self._processor = processor
        self._model = model
        return processor, model, device

    def _embed_images(
        self,
        *,
        checkpoint_path: str,
        images: Sequence[Any],
        device_preference: str = "auto",
        batch_size: int = 8,
        log_fn=None,
    ) -> Any:
        torch = get_torch()

        processor, model, device = self.ensure_model_loaded(
            checkpoint_path,
            device_preference=device_preference,
            log_fn=log_fn,
        )
        effective_batch_size = max(1, int(batch_size))
        rows = []
        for start in range(0, len(images), effective_batch_size):
            batch = list(images[start : start + effective_batch_size])
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {name: value.to(device) if hasattr(value, "to") else value for name, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                tokens = getattr(outputs, "last_hidden_state", None)
                if tokens is None:
                    raise RuntimeError("DINOv2 embedding model did not return last_hidden_state.")
                pooled = tokens[:, 0]
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            rows.append(pooled)
        return torch.cat(rows, dim=0) if rows else torch.empty((0, 0), device=device)

    def _discover_support_images(self, prototype_dir: str, *, include_labels: Sequence[str] | None = None) -> dict[str, list[Path]]:
        root = Path(prototype_dir).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Prototype directory not found: {root}")

        allowed = {_normalize_label(label) for label in (include_labels or []) if str(label or "").strip()}
        groups: dict[str, list[Path]] = {}
        for child in sorted(root.iterdir(), key=lambda path: path.name.lower()):
            if not child.is_dir():
                continue
            label = str(child.name).strip()
            if not label:
                continue
            if not _label_allowed(label, allowed):
                continue
            images = [
                path
                for path in sorted(child.rglob("*"))
                if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
            ]
            if images:
                groups[label] = images
        if not groups:
            allowed_text = f" for labels {sorted(allowed)}" if allowed else ""
            raise FileNotFoundError(f"No prototype images found in {root}{allowed_text}.")
        return groups

    def _load_prototypes(
        self,
        *,
        checkpoint_path: str,
        prototype_dir: str,
        include_labels: Sequence[str] | None = None,
        device_preference: str = "auto",
        batch_size: int = 8,
        log_fn=None,
    ) -> dict[str, Any]:
        from PIL import Image

        processor, model, device = self.ensure_model_loaded(
            checkpoint_path,
            device_preference=device_preference,
            log_fn=log_fn,
        )
        del processor, model

        normalized_labels = tuple(sorted({_normalize_label(label) for label in (include_labels or []) if str(label or "").strip()}))
        root = str(Path(prototype_dir).expanduser().resolve())
        cache_key = (str(checkpoint_path or "").strip(), device, root, normalized_labels)
        cached = self._prototype_cache.get(cache_key)
        if cached is not None:
            return cached

        grouped_paths = self._discover_support_images(root, include_labels=include_labels)
        support_labels: list[str] = []
        support_images: list[Any] = []
        for label, paths in grouped_paths.items():
            for path in paths:
                with Image.open(path) as image:
                    support_images.append(image.convert("RGB"))
                support_labels.append(label)

        embeddings = self._embed_images(
            checkpoint_path=checkpoint_path,
            images=support_images,
            device_preference=device_preference,
            batch_size=batch_size,
            log_fn=log_fn,
        )
        torch = get_torch()
        grouped_embeddings: dict[str, list[Any]] = defaultdict(list)
        for label, embedding in zip(support_labels, embeddings):
            grouped_embeddings[label].append(embedding)

        labels: list[str] = []
        counts: list[int] = []
        vectors: list[Any] = []
        for label in sorted(grouped_embeddings, key=lambda item: item.lower()):
            stacked = torch.stack(grouped_embeddings[label], dim=0)
            prototype = torch.nn.functional.normalize(stacked.mean(dim=0), p=2, dim=-1)
            labels.append(label)
            counts.append(int(stacked.shape[0]))
            vectors.append(prototype)

        if not vectors:
            raise RuntimeError(f"Cannot build any DINOv2 prototypes from {root}")

        payload = {
            "labels": labels,
            "counts": counts,
            "matrix": torch.stack(vectors, dim=0),
            "prototype_dir": root,
        }
        self._prototype_cache[cache_key] = payload
        if log_fn is not None:
            log_fn(f"Built {len(labels)} DINOv2 prototype(s) from {root}.")
        return payload

    def classify_crops(
        self,
        *,
        checkpoint_path: str,
        prototype_dir: str,
        images: Sequence[Any],
        include_labels: Sequence[str] | None = None,
        device_preference: str = "auto",
        batch_size: int = 8,
        top_k: int = 3,
        log_fn=None,
    ) -> list[dict[str, Any]]:
        torch = get_torch()

        if not images:
            return []

        prototypes = self._load_prototypes(
            checkpoint_path=checkpoint_path,
            prototype_dir=prototype_dir,
            include_labels=include_labels,
            device_preference=device_preference,
            batch_size=batch_size,
            log_fn=log_fn,
        )
        crop_embeddings = self._embed_images(
            checkpoint_path=checkpoint_path,
            images=images,
            device_preference=device_preference,
            batch_size=batch_size,
            log_fn=log_fn,
        )
        matrix = prototypes["matrix"]
        similarities = crop_embeddings @ matrix.T
        effective_top_k = max(1, min(int(top_k), int(matrix.shape[0])))
        top_scores, top_ids = torch.topk(similarities, k=effective_top_k, dim=-1)
        labels = list(prototypes["labels"])
        counts = list(prototypes["counts"])
        outputs_payload: list[dict[str, Any]] = []
        for row_scores, row_ids in zip(top_scores.detach().cpu().tolist(), top_ids.detach().cpu().tolist()):
            predictions = []
            for score, prototype_id in zip(row_scores, row_ids):
                label = labels[int(prototype_id)]
                predictions.append(
                    {
                        "label": str(label),
                        "similarity": float(score),
                        "support_count": int(counts[int(prototype_id)]),
                        "prototype_id": int(prototype_id),
                    }
                )
            top_prediction = predictions[0] if predictions else {"label": "", "similarity": 0.0, "support_count": 0, "prototype_id": -1}
            outputs_payload.append(
                {
                    "label": str(top_prediction.get("label") or ""),
                    "similarity": float(top_prediction.get("similarity") or 0.0),
                    "support_count": int(top_prediction.get("support_count") or 0),
                    "prototype_id": int(top_prediction.get("prototype_id") or -1),
                    "top_predictions": predictions,
                    "prototype_labels": labels,
                }
            )
        return outputs_payload
