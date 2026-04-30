from __future__ import annotations

import sys

from .train import main as train_main


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--eval-only" not in args:
        args.append("--eval-only")
    return train_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
