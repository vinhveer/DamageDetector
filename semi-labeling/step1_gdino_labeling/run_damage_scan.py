#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def _resolve_repo_root() -> Path:
    current = Path(__file__).resolve()
    # .../DamageDetector/semi-labeling/step1_gdino_labeling/run_damage_scan.py
    # parents[0]=step1_gdino_labeling, parents[1]=semi-labeling, parents[2]=DamageDetector
    for candidate in current.parents:
        if (candidate / "object_detection").exists() and (candidate / "tools").exists():
            return candidate
    return current.parents[2]


def main(argv: list[str] | None = None) -> int:
    repo_root = _resolve_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from object_detection.damage_scan.cli import main as damage_scan_main

    return int(damage_scan_main(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
