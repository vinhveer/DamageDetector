from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


def resolve_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "object_detection").exists() and (candidate / "tools").exists():
            return candidate
    return current.parents[2]


REPO_ROOT = resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class DinoV2Embedder:
    def __init__(self, *, model_name: str, device: str) -> None:
        from torch_runtime import describe_device_fallback, get_torch, select_device_str
        from transformers import AutoImageProcessor, AutoModel

        self.torch = get_torch()
        self.device = select_device_str(device)
        fallback = describe_device_fallback(device, self.device)
        if fallback:
            print(fallback, flush=True)

        local_files_only = Path(model_name).expanduser().exists()
        source = "local folder" if local_files_only else "model id/cache"
        print(f"Loading DINOv2 from {source}: {model_name}", flush=True)
        self.processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=local_files_only, use_fast=False)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.model.to(self.device)
        self.model.eval()
        self.model_name = model_name
        self.dim = int(getattr(getattr(self.model, "config", None), "hidden_size", 0) or 0)

    def embed(self, images: Sequence[Image.Image], *, batch_size: int) -> np.ndarray:
        rows: list[np.ndarray] = []
        effective_batch_size = max(1, int(batch_size))
        for start in range(0, len(images), effective_batch_size):
            batch = list(images[start : start + effective_batch_size])
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
                tokens = getattr(outputs, "last_hidden_state", None)
                if tokens is None:
                    raise RuntimeError("DINOv2 model did not return last_hidden_state.")
                pooled = tokens[:, 0]
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=-1)
            rows.append(pooled.detach().cpu().numpy().astype(np.float32, copy=False))
        if not rows:
            dim = max(0, int(self.dim))
            return np.empty((0, dim), dtype=np.float32)
        embeddings = np.concatenate(rows, axis=0).astype(np.float32, copy=False)
        if self.dim <= 0 and embeddings.ndim == 2:
            self.dim = int(embeddings.shape[1])
        return embeddings
