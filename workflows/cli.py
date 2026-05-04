from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run_json(cmd: list[str], *, dry_run: bool = False) -> dict[str, Any]:
    if dry_run:
        return {"dry_run": True, "command": cmd}
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Command did not return JSON: {' '.join(cmd)}\n{proc.stdout}") from exc


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_image_max_dim(image_path: str) -> int:
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
            return max(int(width), int(height))
    except Exception:
        return 0


def _append_roi(cmd: list[str], roi: list[int] | None) -> None:
    if roi:
        cmd.extend(["--roi", *(str(int(v)) for v in roi)])


def _append_flag(cmd: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        cmd.append(flag)


def _dino_cmd(args: argparse.Namespace, *, dino_output_dir: Path) -> list[str]:
    max_dim = _read_image_max_dim(args.image)
    use_recursive = args.dino_mode == "recursive" or (
        args.dino_mode == "auto" and bool(args.tile_large_images) and max_dim > int(args.tile_trigger_px)
    )
    command = "recursive-detect" if use_recursive else "predict"
    cmd = [
        sys.executable,
        "-m",
        "object_detection.dino",
        command,
        "--image",
        args.image,
        "--checkpoint",
        args.dino_checkpoint,
        "--config-id",
        args.dino_config_id,
        "--queries",
        args.queries,
        "--box-threshold",
        str(float(args.box_threshold)),
        "--text-threshold",
        str(float(args.text_threshold)),
        "--max-dets",
        str(int(args.max_dets)),
        "--device",
        args.device,
        "--output-dir",
        str(dino_output_dir),
    ]
    _append_roi(cmd, args.roi)
    if use_recursive:
        for label in args.target_label or []:
            cmd.extend(["--target-label", label])
        cmd.extend(["--max-depth", str(int(args.max_depth)), "--min-box-px", str(int(args.min_box_px))])
    return cmd


def _sam_cmd(args: argparse.Namespace, *, sam_output_dir: Path, boxes_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "segmentation.sam.no_finetune",
        "segment-boxes",
        "--image",
        args.image,
        "--boxes-json",
        str(boxes_path),
        "--checkpoint",
        args.sam_checkpoint,
        "--sam-model-type",
        args.sam_model_type,
        "--min-area",
        str(int(args.min_area)),
        "--dilate",
        str(int(args.dilate)),
        "--device",
        args.device,
        "--output-dir",
        str(sam_output_dir),
    ]
    _append_flag(cmd, bool(args.invert_mask), "--invert-mask")
    _append_roi(cmd, args.roi)
    return cmd


def run_sam_dino(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    dino_output_dir = output_dir / "dino"
    sam_output_dir = output_dir / "sam"
    boxes_path = output_dir / "boxes.json"

    dino_cmd = _dino_cmd(args, dino_output_dir=dino_output_dir)
    sam_cmd = _sam_cmd(args, sam_output_dir=sam_output_dir, boxes_path=boxes_path)
    if bool(args.dry_run):
        return {
            "dry_run": True,
            "image_path": args.image,
            "output_dir": str(output_dir),
            "boxes_json": str(boxes_path),
            "commands": {
                "dino": dino_cmd,
                "sam": sam_cmd,
            },
        }

    dino_result = _run_json(dino_cmd, dry_run=bool(args.dry_run))
    boxes = list(dino_result.get("display_detections") or dino_result.get("detections") or [])
    _write_json(boxes_path, boxes)

    if not boxes:
        return {
            "image_path": args.image,
            "output_dir": str(output_dir),
            "boxes_json": str(boxes_path),
            "dino": dino_result,
            "sam": None,
            "detections": [],
        }

    sam_result = _run_json(sam_cmd, dry_run=bool(args.dry_run))
    return {
        "image_path": args.image,
        "output_dir": str(output_dir),
        "boxes_json": str(boxes_path),
        "dino": dino_result,
        "sam": sam_result,
        "detections": list(sam_result.get("detections") or []),
        "mask_path": sam_result.get("mask_path"),
        "overlay_path": sam_result.get("overlay_path"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m damage_detect", description="CLI-only damage detection pipelines.")
    parser.add_argument("--pretty", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sam_dino = sub.add_parser("sam-dino", help="Run DINO boxes, then SAM box prompting, through CLIs only.")
    sam_dino.add_argument("--image", required=True)
    sam_dino.add_argument("--output-dir", default="results_damage_detect")
    sam_dino.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    sam_dino.add_argument("--roi", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"))
    sam_dino.add_argument("--dry-run", action="store_true", help="Print generated commands without running models.")

    sam_dino.add_argument("--dino-checkpoint", required=True)
    sam_dino.add_argument("--dino-config-id", default="auto")
    sam_dino.add_argument("--queries", default="crack")
    sam_dino.add_argument("--target-label", action="append", dest="target_label")
    sam_dino.add_argument("--box-threshold", type=float, default=0.25)
    sam_dino.add_argument("--text-threshold", type=float, default=0.25)
    sam_dino.add_argument("--max-dets", type=int, default=20)
    sam_dino.add_argument("--dino-mode", choices=["auto", "predict", "recursive"], default="auto")
    sam_dino.add_argument("--tile-large-images", action=argparse.BooleanOptionalAction, default=True)
    sam_dino.add_argument("--tile-trigger-px", type=int, default=512)
    sam_dino.add_argument("--max-depth", type=int, default=3)
    sam_dino.add_argument("--min-box-px", type=int, default=48)

    sam_dino.add_argument("--sam-checkpoint", required=True)
    sam_dino.add_argument("--sam-model-type", default="auto", choices=["auto", "vit_b", "vit_l", "vit_h"])
    sam_dino.add_argument("--invert-mask", action="store_true")
    sam_dino.add_argument("--min-area", type=int, default=0)
    sam_dino.add_argument("--dilate", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "sam-dino":
        result = run_sam_dino(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2 if bool(args.pretty) else None)
    sys.stdout.write("\n")
    return 0
