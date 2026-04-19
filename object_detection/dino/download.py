from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable


def download_hf_model(repo_id: str, out_dir: str | os.PathLike[str], *, log_fn: Callable[[str], None] | None = None) -> str:
    from huggingface_hub import snapshot_download

    repo_id = str(repo_id or "").strip()
    if not repo_id:
        raise ValueError("repo_id is required")

    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if log_fn is not None:
        log_fn(f"Downloading {repo_id} -> {out}")
        bad_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if bad_proxy:
            log_fn(f"HTTPS_PROXY={bad_proxy}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(out),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    if log_fn is not None:
        log_fn("Download complete.")
    return str(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download GroundingDINO model repo for offline use.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    download_hf_model(args.model, args.out, log_fn=lambda msg: print(msg, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
