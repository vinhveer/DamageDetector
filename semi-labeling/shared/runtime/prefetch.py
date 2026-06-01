"""Ordered, bounded prefetch over a sequence of items.

GPU inference loops (OpenCLIP semantic, DINOv2 embed) spend most of their wall
time single-threaded decoding/cropping images from disk while the GPU sits
idle. ``ordered_prefetch`` runs the CPU/disk-bound ``load_fn`` in a small thread
pool so the next items are ready by the time the GPU finishes the current batch.

Guarantees:
- Results are yielded in the SAME order as ``items`` (so DB writes / logging
  indices stay stable and reproducible).
- At most ``max_inflight`` ``load_fn`` calls are in flight at once, so memory is
  bounded regardless of how many items there are (e.g. 150k crops).
- Per-item exceptions are captured and returned, not raised, so one unreadable
  crop does not abort the whole run (callers record it as a skip/error).
- ``num_workers <= 0`` falls back to a synchronous generator (identical output,
  no threads) — i.e. the original behaviour.

PIL/JPEG decode and disk I/O release the GIL, so threads give real overlap here
without needing multiprocessing.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Iterator, Tuple, TypeVar

T = TypeVar("T")
R = TypeVar("R")

# (item, result, error) — exactly one of result/error is meaningful.
PrefetchItem = Tuple[T, "R | None", "Exception | None"]


def ordered_prefetch(
    items: Iterable[T],
    load_fn: Callable[[T], R],
    *,
    num_workers: int,
    max_inflight: int | None = None,
) -> Iterator[PrefetchItem]:
    """Yield ``(item, result, error)`` in input order.

    ``load_fn`` runs in worker threads when ``num_workers > 0``. Exceptions from
    ``load_fn`` are captured per item and returned as the third tuple element.
    """
    workers = int(num_workers or 0)

    if workers <= 0:
        for item in items:
            try:
                yield item, load_fn(item), None
            except Exception as exc:  # noqa: BLE001 - surfaced to caller
                yield item, None, exc
        return

    inflight = int(max_inflight) if max_inflight else workers * 2
    inflight = max(inflight, workers, 1)

    iterator = iter(items)
    pending: deque[tuple[T, Any]] = deque()

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="prefetch") as executor:
        # Prime the window.
        for _ in range(inflight):
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending.append((item, executor.submit(load_fn, item)))

        while pending:
            item, future = pending.popleft()
            try:
                yield item, future.result(), None
            except Exception as exc:  # noqa: BLE001 - surfaced to caller
                yield item, None, exc
            # Refill one slot to keep the window full.
            try:
                nxt = next(iterator)
            except StopIteration:
                continue
            pending.append((nxt, executor.submit(load_fn, nxt)))
