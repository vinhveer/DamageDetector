"""C6: per-label calibration from review statistics. stdlib only."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

_DEFAULT_BASE = 0.40
_ERROR_FLOOR = 0.10
_DELTA_CAP = 0.15


@dataclass(frozen=True)
class LabelCalibrationStats:
    label: str
    total_reviewed: int
    confirmed_count: int
    relabeled_count: int
    rejected_count: int

    @property
    def error_rate(self) -> float:
        if self.total_reviewed <= 0:
            return 0.0
        return (int(self.relabeled_count) + int(self.rejected_count)) / int(self.total_reviewed)

    @property
    def suggested_threshold_delta(self) -> float:
        er = self.error_rate
        if er <= _ERROR_FLOOR:
            return 0.0
        return min(_DELTA_CAP, (er - _ERROR_FLOOR) * 0.5)


def compute_calibration(
    stats: Iterable[LabelCalibrationStats], base_thresholds: dict[str, float] | None = None
) -> dict:
    base_thresholds = base_thresholds or {}
    label_thresholds: dict[str, dict] = {}
    stat_summary: dict[str, dict] = {}
    for stat in stats:
        if stat.total_reviewed <= 0:  # exclude unreviewed labels
            continue
        base = float(base_thresholds.get(stat.label, _DEFAULT_BASE))
        delta = stat.suggested_threshold_delta
        label_thresholds[stat.label] = {"base": base, "delta": delta, "effective": base + delta}
        stat_summary[stat.label] = {"total_reviewed": int(stat.total_reviewed), "error_rate": stat.error_rate}
    return {
        "computed_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "label_thresholds": label_thresholds,
        "stats": stat_summary,
    }


def write_calibration(path: Path, calib: dict) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calib, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def load_calibration(path: Path) -> dict:
    """Return the calibration mapping, or {} when the file is missing/unreadable."""
    try:
        return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def effective_threshold(calib: dict, label: str, base: float = _DEFAULT_BASE) -> float:
    """base + delta for the label, or base when absent/unreadable (Requirement 6.5, 6.10)."""
    entry = (calib or {}).get("label_thresholds", {}).get(label)
    if not entry:
        return float(base)
    return float(entry.get("base", base)) + float(entry.get("delta", 0.0))
