from __future__ import annotations

# The SAM runtime worker implementation lives in segmentation.sam.runtime.worker.
# Keep this import shim so existing CLIs that spawn
# `segmentation.sam.no_finetune.worker` continue to work.

from ..runtime.worker import main


if __name__ == "__main__":
    raise SystemExit(main())
