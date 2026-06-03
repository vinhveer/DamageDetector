from __future__ import annotations

__all__ = ["main", "run"]


def __getattr__(name: str):
    if name in {"main", "run"}:
        from .app import main, run

        return {"main": main, "run": run}[name]
    raise AttributeError(name)
