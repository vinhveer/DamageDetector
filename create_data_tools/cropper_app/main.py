from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from create_data_tools.cropper_app.app import run

    run()


if __name__ == "__main__":
    main()

