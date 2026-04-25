from __future__ import annotations

import re
from typing import Any

from torch_runtime import describe_device_fallback, get_torch, has_cuda, has_mps, select_device_str


def load_yolo_class() -> Any:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "Cannot import ultralytics. Install it first with: pip install ultralytics"
        ) from exc
    return YOLO


def _available_cuda_devices() -> list[int]:
    if not has_cuda():
        return []
    try:
        torch = get_torch()
        return list(range(int(torch.cuda.device_count())))
    except Exception:
        return []


def _normalize_explicit_cuda_spec(preference: str) -> str | None:
    raw = str(preference or "").strip().lower()
    if not raw:
        return None
    if re.fullmatch(r"\d+(?:\s*,\s*\d+)*", raw):
        return ",".join(part.strip() for part in raw.split(","))
    if raw.startswith("cuda:"):
        tail = raw[5:].strip()
        if re.fullmatch(r"\d+(?:\s*,\s*\d+)*", tail):
            return ",".join(part.strip() for part in tail.split(","))
    return None


def resolve_device(preference: str, *, num_gpus: int = 0) -> str:
    explicit_cuda = _normalize_explicit_cuda_spec(preference)
    if explicit_cuda is not None:
        return explicit_cuda

    pref = str(preference or "auto").strip().lower()
    requested_gpus = int(num_gpus or 0)
    available_cuda = _available_cuda_devices()

    if pref in {"auto", "cuda"} and available_cuda:
        if requested_gpus <= 0:
            requested_gpus = len(available_cuda)
        requested_gpus = max(1, min(requested_gpus, len(available_cuda)))
        selected = available_cuda[:requested_gpus]
        return ",".join(str(device_id) for device_id in selected)

    if pref == "mps":
        if has_mps():
            return "mps"
        fallback = describe_device_fallback(preference, "cuda" if available_cuda else "cpu")
        if fallback:
            print(fallback)
        return "cuda" if available_cuda else "cpu"

    resolved = select_device_str(preference)
    fallback = describe_device_fallback(preference, resolved)
    if fallback:
        print(fallback)
    return resolved


def configure_yolo_dataloader(
    *,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
) -> None:
    from ultralytics import data as ultralytics_data  # type: ignore
    from ultralytics.data import build as data_build  # type: ignore
    from ultralytics.models.yolo.detect import train as detect_train  # type: ignore
    from ultralytics.models.yolo.detect import val as detect_val  # type: ignore

    def _build_dataloader(
        dataset: Any,
        batch: int,
        workers: int,
        shuffle: bool = True,
        rank: int = -1,
        drop_last: bool = False,
        loader_pin_memory: bool = True,
    ) -> Any:
        batch = min(int(batch), len(dataset))
        nd = data_build.torch.cuda.device_count()
        nw = min(data_build.os.cpu_count() // max(nd, 1), int(workers))
        sampler = (
            None
            if rank == -1
            else data_build.distributed.DistributedSampler(dataset, shuffle=shuffle)
            if shuffle
            else data_build.ContiguousDistributedSampler(dataset)
        )
        generator = data_build.torch.Generator()
        generator.manual_seed(6148914691236517205 + data_build.RANK)

        loader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": batch,
            "shuffle": shuffle and sampler is None,
            "num_workers": nw,
            "sampler": sampler,
            "pin_memory": nd > 0 and (pin_memory if pin_memory is not None else bool(loader_pin_memory)),
            "collate_fn": getattr(dataset, "collate_fn", None),
            "worker_init_fn": data_build.seed_worker,
            "generator": generator,
            "drop_last": bool(drop_last) and len(dataset) % batch != 0,
        }
        effective_prefetch = int(prefetch_factor) if prefetch_factor is not None else 4
        if nw > 0 and effective_prefetch > 0:
            loader_kwargs["prefetch_factor"] = int(effective_prefetch)
        if nw > 0 and persistent_workers is not None:
            loader_kwargs["persistent_workers"] = bool(persistent_workers)
        return data_build.InfiniteDataLoader(**loader_kwargs)

    data_build.build_dataloader = _build_dataloader
    ultralytics_data.build_dataloader = _build_dataloader
    detect_train.build_dataloader = _build_dataloader
    detect_val.build_dataloader = _build_dataloader
