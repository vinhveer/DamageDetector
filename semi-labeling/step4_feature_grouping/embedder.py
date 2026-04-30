from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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
        from torch_runtime import describe_device_fallback, select_device_str
        from transformers import AutoImageProcessor, AutoModel

        self.device = select_device_str(device)
        fallback = describe_device_fallback(device, self.device)
        if fallback:
            print(fallback, flush=True)
        local_files_only = Path(model_name).expanduser().exists()
        self.processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.model.to(self.device)
        self.model.eval()
        self.model_name = model_name

    def embed(self, images: list[Image.Image], *, batch_size: int) -> Any:
        import torch

        rows = []
        effective_batch_size = max(1, int(batch_size))
        for start in range(0, len(images), effective_batch_size):
            batch = images[start : start + effective_batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = self.model(**inputs)
                tokens = getattr(outputs, "last_hidden_state", None)
                if tokens is None:
                    raise RuntimeError("DINOv2 model did not return last_hidden_state.")
                pooled = tokens[:, 0]
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            rows.append(pooled.detach().cpu())
        return torch.cat(rows, dim=0) if rows else torch.empty((0, 0))
