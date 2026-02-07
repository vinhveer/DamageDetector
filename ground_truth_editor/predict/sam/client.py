from __future__ import annotations

from predict.process_client import JsonServiceProcess


_SAM: JsonServiceProcess | None = None


def get_sam_service() -> JsonServiceProcess:
    global _SAM
    if _SAM is None:
        _SAM = JsonServiceProcess(module="predict.sam.worker")
    return _SAM

