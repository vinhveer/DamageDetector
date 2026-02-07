from __future__ import annotations

from predict.process_client import JsonServiceProcess


_UNET: JsonServiceProcess | None = None


def get_unet_service() -> JsonServiceProcess:
    global _UNET
    if _UNET is None:
        _UNET = JsonServiceProcess(module="predict.unet.worker")
    return _UNET

