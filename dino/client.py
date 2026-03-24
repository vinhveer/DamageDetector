from __future__ import annotations

from inference_api.process_client import JsonServiceProcess


_DINO: JsonServiceProcess | None = None


def get_dino_service() -> JsonServiceProcess:
    global _DINO
    if _DINO is None:
        _DINO = JsonServiceProcess(module="dino.worker")
    return _DINO
