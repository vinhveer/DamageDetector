"""Box annotation I/O — pure functions, no Qt dependency (easy to test).

On-disk schema for both the input ``abc.json`` and the saved ``abc.jsonm`` is a
flat JSON list of detection rows::

    [
        {"box": [x1, y1, x2, y2], "label": "crack", "score": 0.71},
        {"box": [x1, y1, x2, y2], "label": "mold",  "score": 1.0}
    ]

``box`` is absolute xyxy in image pixel coordinates. ``score`` defaults to 1.0
for boxes added by hand. ``label`` defaults to an empty string.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BoxData:
    box: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    label: str = ""
    score: float = 1.0

    def normalized(self) -> "BoxData":
        x1, y1, x2, y2 = self.box
        return BoxData(
            box=[min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
            label=self.label,
            score=self.score,
        )


def load_boxes(path: str | Path) -> list[BoxData]:
    """Read a flat box list from ``path``.

    Items missing a valid 4-element ``box`` are skipped. Accepts either a bare
    list or a dict with a ``boxes``/``detections`` key (defensive).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("boxes", "detections", "rows"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = []
    if not isinstance(data, list):
        return []

    boxes: list[BoxData] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        raw = item.get("box")
        if not (isinstance(raw, (list, tuple)) and len(raw) == 4):
            continue
        try:
            box = [float(v) for v in raw]
        except (TypeError, ValueError):
            continue
        label = str(item.get("label") or "")
        try:
            score = float(item.get("score", 1.0))
        except (TypeError, ValueError):
            score = 1.0
        boxes.append(BoxData(box=box, label=label, score=score).normalized())
    return boxes


def save_boxes(path: str | Path, boxes: list[BoxData], *, ndigits: int = 2) -> None:
    """Write ``boxes`` to ``path`` as a flat JSON list (same schema as input)."""
    payload = [
        {
            "box": [round(float(v), ndigits) for v in b.normalized().box],
            "label": b.label,
            "score": round(float(b.score), 4),
        }
        for b in boxes
    ]
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
