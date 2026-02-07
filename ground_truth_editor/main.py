from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        # Keep the script directory (ground_truth_editor/) as sys.path[0] so `from ui import run` works.
        sys.path.insert(1, str(repo_root))


_ensure_repo_root_on_path()

from ui import run  # noqa: E402


if __name__ == "__main__":
    run()
