#!/usr/bin/env python3
"""GDINO scan CLI — delegates to object_detection.damage_scan.cli.

All flags (recall/tiled options) are defined there; this just makes the repo
root importable and forwards argv.
"""
from __future__ import annotations

import sys

from shared.runtime import bootstrap

bootstrap.ensure_repo_root_on_path()

from object_detection.damage_scan.cli import main as _damage_scan_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return int(_damage_scan_main(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
