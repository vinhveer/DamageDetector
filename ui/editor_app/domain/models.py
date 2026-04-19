from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FINAL_JOB_STATUSES = {"done", "failed", "cancelled"}


@dataclass(frozen=True)
class RunContext:
    run_id: str
    workflow: str
    scope: str
    run_dir: Path
    output_dir: Path
    data_dir: Path
    created_at: str


@dataclass
class JobRecord:
    job_id: str
    run_id: str
    workflow: str
    scope: str
    status: str
    task_group: str = ""
    segmentation_model: str = ""
    detection_model: str = ""
    resolved_workflow: str = ""
    image_path: str | None = None
    image_paths: list[str] = field(default_factory=list)
    run_dir: str = ""
    output_dir: str = ""
    created_at: str = ""
    request_data: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    partial_payloads: list[dict[str, Any]] = field(default_factory=list)
    final_payload: dict[str, Any] | None = None
    error: str | None = None

    def is_finished(self) -> bool:
        return self.status in FINAL_JOB_STATUSES


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    workflow: str
    status: str
    created_at: str
    run_dir: str
    task_group: str = ""
    segmentation_model: str = ""
    detection_model: str = ""
    request_path: str | None = None
    result_path: str | None = None
    log_path: str | None = None
