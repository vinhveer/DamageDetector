from __future__ import annotations

import csv
import json
from pathlib import Path

from prompts import PromptGroup


def rel_image_key(input_dir: Path, image_path: Path) -> str:
    return image_path.relative_to(input_dir).as_posix()


def write_manifest(
    *,
    manifest_path: Path,
    run_id: str,
    prompt_groups: list[PromptGroup],
    request_names: list[str],
    input_dir: Path,
    output_dir: Path,
    checkpoint: str,
    total_images: int,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "run_id": run_id,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "checkpoint": checkpoint,
        "request_names": request_names,
        "engine_name": "groundingdino",
        "prompt_groups": [
            {
                "prompt_group_id": group.group_id,
                "name": group.name,
                "slug": group.slug,
                "prompt_text": group.prompt_text,
                "queries": list(group.queries),
            }
            for group in prompt_groups
        ],
        "image_count": int(total_images),
    }
    manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary_csv(summary_path: Path, summary_rows: list[tuple]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "prompt_group_id",
                "prompt_group_name",
                "image_rel_path",
                "status",
                "mode",
                "detection_count",
                "artifact_dir",
                "error_type",
                "error_message",
            ]
        )
        writer.writerows(summary_rows)


def write_readme(readme_path: Path) -> None:
    text = (
        "semi-labeling output\n"
        "\n"
        "- boxes.sqlite3: aggregated box database.\n"
        "- prompt_groups.json: prompt set manifest for this pass.\n"
        "- image_runs.csv: per-image status summary.\n"
        "- This pass stores box coordinates in SQLite only. No overlay or box CSV exports are written.\n"
        "\n"
        "Note: in this repository, prompt-based `dino` requests are served by the GroundingDINO engine.\n"
        "The SQLite table `request_aliases` records this mapping explicitly.\n"
    )
    readme_path.write_text(text, encoding="utf-8")
