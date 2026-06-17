from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class JobKind(str, Enum):
    detect = "detect"
    segment = "segment"
    export = "export"


@dataclass
class JobSpec:
    id: str = field(default_factory=lambda: f"J_{uuid4().hex[:8]}")
    kind: JobKind = JobKind.detect
    label: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.queued
    progress: float = 0.0
    message: str = ""
    error: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: Any | None = None


@dataclass
class JobUpdate:
    progress: float | None = None
    message: str | None = None
    log_line: str | None = None
