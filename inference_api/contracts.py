from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DetectionResult:
    label: str
    score: float = 0.0
    box: list[float] | None = None
    mask_path: str | None = None
    overlay_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "label": self.label,
            "score": float(self.score),
        }
        if self.box is not None:
            data["box"] = list(self.box)
        if self.mask_path:
            data["mask_path"] = self.mask_path
        if self.overlay_path:
            data["overlay_path"] = self.overlay_path
        data.update(dict(self.extra))
        return data


@dataclass(frozen=True)
class InferenceRequest:
    workflow: str
    image_path: str | None = None
    image_paths: list[str] | None = None
    roi_box: tuple[int, int, int, int] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    client_tag: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class InferenceResult:
    job_id: str
    workflow: str
    payload: dict[str, Any] = field(default_factory=dict)
    detections: list[DetectionResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.payload)
        if self.detections and "detections" not in data:
            data["detections"] = [det.to_dict() for det in self.detections]
        return data


@dataclass(frozen=True)
class JobSnapshot:
    job_id: str
    workflow: str
    status: str
    request: InferenceRequest
    error: str | None = None


@dataclass(frozen=True)
class JobEvent:
    type: str
    job_id: str
    workflow: str
    message: str | None = None
    progress: float | None = None
    result: InferenceResult | None = None
    error: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
