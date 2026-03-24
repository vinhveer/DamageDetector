from __future__ import annotations

from .contracts import DetectionResult, InferenceRequest, InferenceResult, JobEvent, JobSnapshot
from .prediction_models import PredictionConfig, ResolvedWorkflow

__all__ = [
    "DetectionResult",
    "InferenceApi",
    "InferenceRequest",
    "InferenceResult",
    "JobEvent",
    "JobSnapshot",
    "PredictionConfig",
    "ResolvedWorkflow",
    "get_inference_api",
]


def __getattr__(name: str):
    if name in {"InferenceApi", "get_inference_api"}:
        from .api import InferenceApi, get_inference_api

        if name == "InferenceApi":
            return InferenceApi
        return get_inference_api
    raise AttributeError(name)
