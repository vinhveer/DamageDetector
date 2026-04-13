from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from torch_runtime import describe_device_fallback, get_torch, select_device_str


def load_cached_embeddings(cache_path: Path, image_paths: list[Path], checkpoint: str) -> np.ndarray | None:
    if not cache_path.is_file():
        return None
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    expected_paths = [str(path) for path in image_paths]
    if payload.get("checkpoint") != checkpoint:
        return None
    if payload.get("image_paths") != expected_paths:
        return None
    embeddings = np.asarray(payload.get("embeddings"))
    if embeddings.ndim != 2 or embeddings.shape[0] != len(image_paths):
        return None
    print(f"Loaded cached embeddings from {cache_path}")
    return embeddings


def save_cached_embeddings(cache_path: Path, image_paths: list[Path], checkpoint: str, embeddings: np.ndarray) -> None:
    payload = {
        "checkpoint": checkpoint,
        "image_paths": [str(path) for path in image_paths],
        "embeddings": np.asarray(embeddings, dtype=np.float32),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def embed_images(
    image_paths: list[Path],
    checkpoint: str,
    batch_size: int,
    device_preference: str,
    cache_path: Path,
) -> np.ndarray:
    cached_checkpoint = str(checkpoint).strip()
    cached = load_cached_embeddings(cache_path, image_paths, cached_checkpoint)
    if cached is not None:
        return cached

    from transformers import AutoImageProcessor, AutoModel

    torch = get_torch()
    device = select_device_str(device_preference)
    fallback = describe_device_fallback(device_preference, device)
    if fallback is not None:
        print(fallback)

    checkpoint_path = str(checkpoint).strip()
    local_files_only = Path(checkpoint_path).exists()
    print(f"Loading DINOv2 from {'local folder' if local_files_only else 'model id'}: {checkpoint_path}")
    processor = AutoImageProcessor.from_pretrained(checkpoint_path, local_files_only=local_files_only, use_fast=False)
    model = AutoModel.from_pretrained(checkpoint_path, local_files_only=local_files_only)
    model.to(device)
    model.eval()

    rows: list[np.ndarray] = []
    effective_batch = max(1, int(batch_size))
    for start in tqdm(range(0, len(image_paths), effective_batch), desc="Embedding images", unit="batch"):
        batch_paths = image_paths[start:start + effective_batch]
        batch_images = []
        for path in batch_paths:
            with Image.open(path) as image:
                batch_images.append(image.convert("RGB"))
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {name: value.to(device) if hasattr(value, "to") else value for name, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
            tokens = getattr(outputs, "last_hidden_state", None)
            if tokens is None:
                raise RuntimeError("DINOv2 model did not return last_hidden_state.")
            pooled = tokens[:, 0]
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        rows.append(pooled.detach().cpu().numpy().astype(np.float32))

    embeddings = np.concatenate(rows, axis=0) if rows else np.empty((0, 0), dtype=np.float32)
    save_cached_embeddings(cache_path, image_paths, cached_checkpoint, embeddings)
    return embeddings
