#!/usr/bin/env python3
from __future__ import annotations

import sys

from run_feature_grouping import main as run_feature_grouping_main


def has_option(argv: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def main(argv: list[str] | None = None) -> int:
    user_args = list(sys.argv[1:] if argv is None else argv)
    defaults: list[str] = []
    if not has_option(user_args, "--cluster-method"):
        defaults.extend(["--cluster-method", "agglomerative"])
    if not has_option(user_args, "--agglomerative-distance-threshold"):
        defaults.extend(["--agglomerative-distance-threshold", "0.38"])
    if not has_option(user_args, "--model-name"):
        defaults.extend(["--model-name", "facebook/dinov2-small"])
    if not has_option(user_args, "--batch-size"):
        defaults.extend(["--batch-size", "32"])
    return run_feature_grouping_main(defaults + user_args)


if __name__ == "__main__":
    raise SystemExit(main())
