from __future__ import annotations

import argparse
from pathlib import Path

from object_detection.datasets import build_stable_dino_overrides, load_detection_dataset
from torch_runtime import select_device_str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m object_detection.stable_dino.train",
        description="Train StableDINO with shared dataset helpers.",
    )
    parser.add_argument("--dataset", required=True, help="Path to shared detection dataset manifest")
    parser.add_argument(
        "--config-file",
        default="object_detection/stable_dino/projects/stabledino/configs/damage_detector_stabledino_r50_4scale_12ep.py",
        help="StableDINO LazyConfig file",
    )
    parser.add_argument("--output-dir", default="object_detection/stable_dino/train")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--augmentation-profile", default="balanced", choices=["light", "balanced", "aggressive"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0)
    parser.add_argument("--dist-url", default="auto")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Extra LazyConfig overrides")
    return parser


def _worker_main(args: argparse.Namespace) -> None:
    from object_detection.stable_dino.tools import train_net

    manifest = load_detection_dataset(args.dataset)
    resolved_device = select_device_str(args.device)
    dataset_prefix = f"damage_detector_{manifest.yaml_path.stem}"
    cache_root = Path(args.output_dir).expanduser().resolve() / "dataset_cache"
    overrides = build_stable_dino_overrides(
        manifest,
        dataset_name_prefix=dataset_prefix,
        cache_root=cache_root,
        augmentation_profile=args.augmentation_profile,
        image_size=int(args.imgsz),
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        device=resolved_device,
        output_dir=str(Path(args.output_dir).expanduser().resolve()),
        init_checkpoint=str(args.init_checkpoint or "").strip() or None,
    )
    if args.opts:
        overrides.extend(list(args.opts))
    train_args = argparse.Namespace(
        config_file=str(Path(args.config_file).expanduser().resolve()),
        resume=bool(args.resume),
        eval_only=bool(args.eval_only),
        opts=overrides,
        num_gpus=int(args.num_gpus),
        num_machines=int(args.num_machines),
        machine_rank=int(args.machine_rank),
        dist_url=str(args.dist_url),
    )
    train_net.main(train_args)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    from detectron2.engine import launch

    launch(
        _worker_main,
        int(args.num_gpus),
        num_machines=int(args.num_machines),
        machine_rank=int(args.machine_rank),
        dist_url=str(args.dist_url),
        args=(args,),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
