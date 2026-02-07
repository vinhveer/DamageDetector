from __future__ import annotations

from predict.process_client import JsonServiceProcess


_DINO: JsonServiceProcess | None = None


def get_dino_service() -> JsonServiceProcess:
    global _DINO
    if _DINO is None:
        _DINO = JsonServiceProcess(module="predict.dino.worker")
    return _DINO

