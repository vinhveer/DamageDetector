from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image


class DinoV2Embedder:
    def __init__(self, *, model_name: str, device: str) -> None:
        from torch_runtime import describe_device_fallback, select_device_str
        from transformers import AutoImageProcessor, AutoModel
        import torch

        self.torch = torch
        self.device = select_device_str(device)
        fb = describe_device_fallback(device, self.device)
        if fb:
            print(fb, flush=True)
        from pathlib import Path as _P
        local_only = _P(model_name).expanduser().exists()
        self.processor = AutoImageProcessor.from_pretrained(
            model_name, local_files_only=local_only, use_fast=False,
        )
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_only)
        self.model.to(self.device).eval()
        self.model_name = model_name

    def embed(self, images: Sequence[Image.Image], *, batch_size: int = 16) -> np.ndarray:
        if not images:
            return np.zeros((0, 0), dtype=np.float32)
        rows: list[np.ndarray] = []
        bs = max(1, int(batch_size))
        for start in range(0, len(images), bs):
            chunk = list(images[start: start + bs])
            inputs = self.processor(images=chunk, return_tensors="pt")
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            with self.torch.inference_mode():
                out = self.model(**inputs)
                tokens = getattr(out, "last_hidden_state", None)
                if tokens is None:
                    raise RuntimeError("DINOv2 returned no last_hidden_state")
                cls = tokens[:, 0]
                cls = self.torch.nn.functional.normalize(cls, p=2, dim=-1)
            rows.append(cls.detach().cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(rows, axis=0)
