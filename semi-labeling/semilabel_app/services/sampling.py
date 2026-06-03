from __future__ import annotations

from collections import defaultdict
from math import isfinite
from array import array
from typing import Sequence

try:
    import numpy as np
except Exception:  # pragma: no cover - fallback for DB-only tests without numpy
    np = None  # type: ignore[assignment]


Vec = Sequence[float]


def decode_vec(blob: bytes | memoryview | None) -> Vec | None:
    if not blob:
        return None
    raw = bytes(blob)
    if np is not None:
        return np.frombuffer(raw, dtype=np.float32).copy()
    values = array("f")
    values.frombytes(raw)
    return values


def _cosine_dist(a: Vec, b: Vec) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return 1.0
    if np is not None:
        return float(1.0 - np.dot(a[:n], b[:n]))
    dot = 0.0
    for idx in range(n):
        dot += float(a[idx]) * float(b[idx])
    return 1.0 - dot


def _fps_select(items: list[dict], k: int) -> list[int]:
    total = len(items)
    if k >= total:
        return [int(item["result_id"]) for item in items]
    if k <= 0:
        return []

    seed_idx = 0
    for idx in range(1, total):
        current = items[idx]
        seed = items[seed_idx]
        if (
            float(current.get("reliability") or 0) < float(seed.get("reliability") or 0)
            or (
                float(current.get("reliability") or 0) == float(seed.get("reliability") or 0)
                and int(current["result_id"]) < int(seed["result_id"])
            )
        ):
            seed_idx = idx

    selected = [seed_idx]
    if np is not None:
        min_dist = np.full(total, np.inf, dtype=np.float64)
    else:
        min_dist = [float("inf")] * total
    seed_vec = items[seed_idx].get("vec")
    if seed_vec is not None:
        for idx, item in enumerate(items):
            vec = item.get("vec")
            min_dist[idx] = _cosine_dist(vec, seed_vec) if vec is not None else np.inf
    min_dist[seed_idx] = -1.0

    while len(selected) < k:
        best = int(np.argmax(min_dist)) if np is not None else max(range(total), key=lambda idx: min_dist[idx])
        if best < 0 or min_dist[best] < 0:
            break
        selected.append(best)
        best_vec = items[best].get("vec")
        min_dist[best] = -1.0
        if best_vec is None:
            continue
        for idx, item in enumerate(items):
            if min_dist[idx] < 0:
                continue
            vec = item.get("vec")
            dist = _cosine_dist(vec, best_vec) if vec is not None else np.inf
            if dist < min_dist[idx]:
                min_dist[idx] = dist

    return [int(items[idx]["result_id"]) for idx in selected]


def select_diverse_sample(rows: list[dict], ratio: float) -> set[int]:
    r = float(ratio) if isfinite(float(ratio or 0)) else 0.0
    if r <= 0 or r >= 1:
        return {int(row["result_id"]) for row in rows}

    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[str(row.get("label") or "unknown")].append(row)

    picked: set[int] = set()
    for items in by_label.values():
        k = max(1, round(len(items) * r))
        picked.update(_fps_select(items, k))
    return picked
