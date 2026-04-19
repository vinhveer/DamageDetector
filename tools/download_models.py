from __future__ import annotations

import argparse
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from object_detection.dino.download import download_hf_model


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    source: str
    relative_output: str
    filename: str | None = None
    description: str = ""


MODEL_CATALOG: dict[str, ModelSpec] = {
    "sam_vit_b": ModelSpec(
        name="sam_vit_b",
        kind="url",
        source="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        relative_output="models/sam",
        filename="sam_vit_b_01ec64.pth",
        description="Segment Anything ViT-B checkpoint.",
    ),
    "sam_vit_l": ModelSpec(
        name="sam_vit_l",
        kind="url",
        source="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        relative_output="models/sam",
        filename="sam_vit_l_0b3195.pth",
        description="Segment Anything ViT-L checkpoint.",
    ),
    "sam_vit_h": ModelSpec(
        name="sam_vit_h",
        kind="url",
        source="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        relative_output="models/sam",
        filename="sam_vit_h_4b8939.pth",
        description="Segment Anything ViT-H checkpoint.",
    ),
    "grounding_dino_base": ModelSpec(
        name="grounding_dino_base",
        kind="hf",
        source="IDEA-Research/grounding-dino-base",
        relative_output="dino/models/grounding-dino-base",
        description="GroundingDINO base model repo from Hugging Face.",
    ),
    "dinov2_small": ModelSpec(
        name="dinov2_small",
        kind="hf",
        source="facebook/dinov2-small",
        relative_output="dino/models/dinov2-small",
        description="DINOv2 small embedding model from Hugging Face.",
    ),
}


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _download_url(url: str, destination: Path, *, force: bool = False, log_fn: Callable[[str], None] | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        if log_fn is not None:
            log_fn(f"Skipping existing file: {destination}")
        return destination
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    if log_fn is not None:
        log_fn(f"Downloading {url} -> {destination}")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as f:
        shutil.copyfileobj(response, f)
    tmp_path.replace(destination)
    if log_fn is not None:
        log_fn("Download complete.")
    return destination


def download_named_models(
    names: list[str] | None,
    *,
    out_dir: str | Path | None = None,
    force: bool = False,
    log_fn: Callable[[str], None] | None = None,
) -> list[str]:
    repo_root = Path(out_dir).expanduser().resolve() if out_dir else _default_repo_root()
    requested = list(MODEL_CATALOG.keys()) if not names or "all" in names else names
    outputs: list[str] = []
    for name in requested:
        spec = MODEL_CATALOG.get(str(name).strip())
        if spec is None:
            raise KeyError(f"Unknown model preset: {name}")
        target_root = repo_root / spec.relative_output
        if spec.kind == "url":
            assert spec.filename
            path = _download_url(spec.source, target_root / spec.filename, force=force, log_fn=log_fn)
            outputs.append(str(path))
        elif spec.kind == "hf":
            path = download_hf_model(spec.source, target_root, log_fn=log_fn)
            outputs.append(str(path))
        else:
            raise ValueError(f"Unsupported model kind: {spec.kind}")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download model assets used by DamageDetector.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List built-in model presets.")

    download = sub.add_parser("download", help="Download one or more built-in model presets.")
    download.add_argument("--name", action="append", dest="names", help="Preset name to download; repeat as needed. Use 'all' for every preset.")
    download.add_argument("--out-dir", default="", help="Destination repo root override. Defaults to the current repo root.")
    download.add_argument("--force", action="store_true", help="Overwrite existing files.")

    custom = sub.add_parser("download-hf", help="Download an arbitrary Hugging Face model repo.")
    custom.add_argument("--repo-id", required=True)
    custom.add_argument("--out", required=True)

    custom_url = sub.add_parser("download-url", help="Download a raw checkpoint file from a URL.")
    custom_url.add_argument("--url", required=True)
    custom_url.add_argument("--out", required=True)
    custom_url.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        for spec in MODEL_CATALOG.values():
            print(f"{spec.name}: {spec.description} -> {spec.relative_output}")
        return 0
    if args.command == "download":
        names = args.names or ["all"]
        download_named_models(
            names,
            out_dir=(args.out_dir or None),
            force=bool(args.force),
            log_fn=lambda msg: print(msg, flush=True),
        )
        return 0
    if args.command == "download-hf":
        download_hf_model(args.repo_id, args.out, log_fn=lambda msg: print(msg, flush=True))
        return 0
    if args.command == "download-url":
        _download_url(args.url, Path(args.out).expanduser().resolve(), force=bool(args.force), log_fn=lambda msg: print(msg, flush=True))
        return 0
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
