from __future__ import annotations

from inference_api.process_client import JsonServiceProcess


_SAM_FT: JsonServiceProcess | None = None


def get_sam_finetune_service() -> JsonServiceProcess:
    global _SAM_FT
    if _SAM_FT is None:
        _SAM_FT = JsonServiceProcess(module="sam_finetune.worker")
    return _SAM_FT
