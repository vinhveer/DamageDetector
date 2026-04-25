from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


def _resolve_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "object_detection").exists() and (candidate / "tools").exists():
            return candidate
    return current.parents[2]


_REPO_ROOT = _resolve_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prompts import load_prompt_groups, parse_request_names
from runner import run_semi_label_job


def _pass_output_dir(root_output_dir: Path, group) -> Path:
    return root_output_dir / group.slug


def main() -> int:
    config = SimpleNamespace(
        input_dir="/Users/nguyenquangvinh/Desktop/Lab/HinhAnh",
        output_dir="/Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling",
        prompt_group=[],
        request_names="dino,groundingdino",
        checkpoint="",
        box_threshold=0.16,
        text_threshold=0.16,
        max_dets=80,
        device="auto",
        recursive_find=False,
        tiled_threshold=512,
        tile_scales="small,medium,large",
        recursive_max_depth=3,
        recursive_min_box_px=48,
        limit=0,
        verbose_logs=False,
    )

    input_dir = Path(config.input_dir).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve()
    if not str(config.input_dir).strip():
        raise ValueError("Set `input_dir` in main.py before running.")
    if not str(config.output_dir).strip():
        raise ValueError("Set `output_dir` in main.py before running.")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_groups = load_prompt_groups(config.prompt_group)
    request_names = parse_request_names(config.request_names)
    exit_code = 0

    for group in prompt_groups:
        pass_output_dir = _pass_output_dir(output_dir, group)
        pass_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Pass {group.group_id}: {group.name} -> {pass_output_dir}", flush=True)
        result = run_semi_label_job(
            args=config,
            input_dir=input_dir,
            output_dir=pass_output_dir,
            prompt_groups=[group],
            request_names=request_names,
        )
        if result != 0:
            exit_code = result

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
