from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from inference_api.process_client import JsonServiceProcess
from torch_runtime import get_torch, has_cuda


@dataclass(frozen=True)
class DinoServiceConfig:
    num_workers: int
    queue_size: int
    batch_size: int
    device_ids: tuple[int, ...]


@dataclass
class _ServiceSlot:
    index: int
    label: str
    process: JsonServiceProcess
    busy: bool = False
    started_calls: int = 0


def _available_cuda_device_ids() -> tuple[int, ...]:
    if not has_cuda():
        return ()
    try:
        torch = get_torch()
        return tuple(range(int(torch.cuda.device_count())))
    except Exception:
        return ()


def _parse_device_ids(device_ids: str | Sequence[int] | None) -> tuple[int, ...]:
    if device_ids is None:
        return ()
    if isinstance(device_ids, str):
        parts = [part.strip() for part in device_ids.split(",") if part.strip()]
        if not parts:
            return ()
        return tuple(int(part) for part in parts)
    return tuple(int(part) for part in device_ids)


def _resolve_service_config(
    *,
    num_workers: int | None = None,
    queue_size: int | None = None,
    batch_size: int | None = None,
    device_ids: str | Sequence[int] | None = None,
) -> DinoServiceConfig:
    available_ids = _available_cuda_device_ids()
    requested_ids = _parse_device_ids(device_ids)

    if requested_ids:
        resolved_device_ids = requested_ids
        resolved_workers = len(requested_ids)
    elif available_ids:
        requested_workers = int(num_workers or 0)
        if requested_workers <= 0:
            requested_workers = len(available_ids)
        requested_workers = max(1, min(requested_workers, len(available_ids)))
        resolved_device_ids = tuple(available_ids[:requested_workers])
        resolved_workers = len(resolved_device_ids)
    else:
        resolved_device_ids = ()
        resolved_workers = max(1, int(num_workers or 1))

    resolved_queue_size = int(queue_size or 0)
    if resolved_queue_size <= 0:
        resolved_queue_size = max(2, resolved_workers * 2)

    resolved_batch_size = max(1, int(batch_size or 1))
    return DinoServiceConfig(
        num_workers=int(resolved_workers),
        queue_size=int(resolved_queue_size),
        batch_size=int(resolved_batch_size),
        device_ids=resolved_device_ids,
    )


class DinoServicePool:
    def __init__(self, *, module: str, config: DinoServiceConfig) -> None:
        self._module = str(module)
        self._config = config
        self._slots: list[_ServiceSlot] = []
        self._cv = threading.Condition()
        self._waiting = 0

        for index in range(int(config.num_workers)):
            device_id = config.device_ids[index] if index < len(config.device_ids) else None
            env = {}
            label = f"worker-{index}"
            if device_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(device_id)
                label = f"gpu-{device_id}"
            process = JsonServiceProcess(module=self._module, env=env or None)
            self._slots.append(_ServiceSlot(index=index, label=label, process=process))

    @property
    def config(self) -> DinoServiceConfig:
        return self._config

    def _decorate_log_fn(self, slot: _ServiceSlot, log_fn: Callable[[str], None] | None) -> Callable[[str], None] | None:
        if log_fn is None:
            return None

        def _log(message: str) -> None:
            log_fn(f"[{slot.label}] {message}")

        return _log

    def _acquire_slot(self) -> _ServiceSlot:
        with self._cv:
            waiting_registered = False
            while True:
                free_slots = [slot for slot in self._slots if not slot.busy]
                if free_slots:
                    slot = min(free_slots, key=lambda item: (item.started_calls, item.index))
                    if waiting_registered:
                        self._waiting = max(0, self._waiting - 1)
                    slot.busy = True
                    slot.started_calls += 1
                    return slot
                if not waiting_registered:
                    if self._waiting >= int(self._config.queue_size):
                        raise RuntimeError(
                            f"DINO service queue is full: waiting={self._waiting}, workers={len(self._slots)}"
                        )
                    self._waiting += 1
                    waiting_registered = True
                self._cv.wait()

    def _release_slot(self, slot: _ServiceSlot) -> None:
        with self._cv:
            slot.busy = False
            self._cv.notify()

    def _reserve_specific_slot(self, slot: _ServiceSlot) -> None:
        with self._cv:
            while slot.busy:
                self._cv.wait()
            slot.busy = True
            slot.started_calls += 1

    def _call_on_slot(
        self,
        slot: _ServiceSlot,
        method: str,
        params: dict[str, Any] | None,
        *,
        log_fn: Callable[[str], None] | None = None,
        stop_checker: Callable[[], bool] | None = None,
        poll_s: float = 0.15,
        timeout_s: float | None = None,
    ) -> Any:
        try:
            return slot.process.call(
                method,
                params,
                log_fn=self._decorate_log_fn(slot, log_fn),
                stop_checker=stop_checker,
                poll_s=poll_s,
                timeout_s=timeout_s,
            )
        finally:
            self._release_slot(slot)

    def _call_single(
        self,
        method: str,
        params: dict[str, Any] | None,
        *,
        log_fn: Callable[[str], None] | None = None,
        stop_checker: Callable[[], bool] | None = None,
        poll_s: float = 0.15,
        timeout_s: float | None = None,
    ) -> Any:
        slot = self._acquire_slot()
        return self._call_on_slot(
            slot,
            method,
            params,
            log_fn=log_fn,
            stop_checker=stop_checker,
            poll_s=poll_s,
            timeout_s=timeout_s,
        )

    def _call_warmup_all(
        self,
        params: dict[str, Any] | None,
        *,
        log_fn: Callable[[str], None] | None = None,
        stop_checker: Callable[[], bool] | None = None,
        poll_s: float = 0.15,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        max_workers = max(1, len(self._slots))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="dino-warmup") as executor:
            futures = [
                executor.submit(
                    self._warmup_slot,
                    slot,
                    params,
                    log_fn=log_fn,
                    stop_checker=stop_checker,
                    poll_s=poll_s,
                    timeout_s=timeout_s,
                )
                for slot in self._slots
            ]
            results = [future.result() for future in futures]
        return {"ok": all(bool((result or {}).get("ok")) for result in results), "workers": len(results)}

    def _warmup_slot(
        self,
        slot: _ServiceSlot,
        params: dict[str, Any] | None,
        *,
        log_fn: Callable[[str], None] | None = None,
        stop_checker: Callable[[], bool] | None = None,
        poll_s: float = 0.15,
        timeout_s: float | None = None,
    ) -> Any:
        self._reserve_specific_slot(slot)
        return self._call_on_slot(
            slot,
            "warmup",
            params,
            log_fn=log_fn,
            stop_checker=stop_checker,
            poll_s=poll_s,
            timeout_s=timeout_s,
        )

    def _call_predict_batch_parallel(
        self,
        params: dict[str, Any] | None,
        *,
        log_fn: Callable[[str], None] | None = None,
        stop_checker: Callable[[], bool] | None = None,
        poll_s: float = 0.15,
        timeout_s: float | None = None,
    ) -> Any:
        payload = dict(params or {})
        image_paths = list(payload.get("image_paths") or [])
        if len(self._slots) <= 1 or len(image_paths) <= 1:
            return self._call_single(
                "predict_batch",
                payload,
                log_fn=log_fn,
                stop_checker=stop_checker,
                poll_s=poll_s,
                timeout_s=timeout_s,
            )

        chunk_size = max(1, int(self._config.batch_size))
        if chunk_size == 1 and len(self._slots) > 1:
            chunk_size = max(1, math.ceil(len(image_paths) / len(self._slots)))

        chunks: list[tuple[int, list[str]]] = []
        for start in range(0, len(image_paths), chunk_size):
            chunks.append((start, image_paths[start : start + chunk_size]))

        def _run_chunk(start_index: int, chunk_paths: list[str]) -> tuple[int, Any]:
            chunk_payload = dict(payload)
            chunk_payload["image_paths"] = list(chunk_paths)
            result = self._call_single(
                "predict_batch",
                chunk_payload,
                log_fn=log_fn,
                stop_checker=stop_checker,
                poll_s=poll_s,
                timeout_s=timeout_s,
            )
            return start_index, result

        max_workers = max(1, min(len(self._slots), len(chunks)))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="dino-batch") as executor:
            futures = [executor.submit(_run_chunk, start, chunk_paths) for start, chunk_paths in chunks]
            chunk_results = [future.result() for future in futures]

        ordered_results: list[dict[str, Any]] = []
        for _start_index, chunk_result in sorted(chunk_results, key=lambda item: item[0]):
            if isinstance(chunk_result, dict) and chunk_result.get("stopped"):
                return {"stopped": True}
            ordered_results.extend(list((chunk_result or {}).get("results") or []))
        return {"batch_done": True, "results": ordered_results}

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        log_fn: Callable[[str], None] | None = None,
        stop_checker: Callable[[], bool] | None = None,
        poll_s: float = 0.15,
        timeout_s: float | None = None,
    ) -> Any:
        normalized = str(method or "").strip().lower()
        if normalized == "warmup":
            return self._call_warmup_all(
                params,
                log_fn=log_fn,
                stop_checker=stop_checker,
                poll_s=poll_s,
                timeout_s=timeout_s,
            )
        if normalized == "predict_batch":
            return self._call_predict_batch_parallel(
                params,
                log_fn=log_fn,
                stop_checker=stop_checker,
                poll_s=poll_s,
                timeout_s=timeout_s,
            )
        return self._call_single(
            method,
            params,
            log_fn=log_fn,
            stop_checker=stop_checker,
            poll_s=poll_s,
            timeout_s=timeout_s,
        )

    def close(self) -> None:
        for slot in self._slots:
            try:
                slot.process.close()
            except Exception:
                pass


_DINO: DinoServicePool | None = None
_DINO_CONFIG: DinoServiceConfig | None = None


def get_dino_service(
    *,
    num_workers: int | None = None,
    queue_size: int | None = None,
    batch_size: int | None = None,
    device_ids: str | Sequence[int] | None = None,
) -> DinoServicePool:
    global _DINO, _DINO_CONFIG
    config = _resolve_service_config(
        num_workers=num_workers,
        queue_size=queue_size,
        batch_size=batch_size,
        device_ids=device_ids,
    )
    if _DINO is None or _DINO_CONFIG != config:
        if _DINO is not None:
            _DINO.close()
        _DINO = DinoServicePool(module="object_detection.dino.worker", config=config)
        _DINO_CONFIG = config
    return _DINO
