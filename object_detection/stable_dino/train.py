from __future__ import annotations

import argparse
import json
import pkgutil
from importlib.machinery import FileFinder
from pathlib import Path

from object_detection.stable_dino.dataset_config import build_stable_dino_overrides, load_stable_dino_dataset_config
from torch_runtime import get_torch, select_device_str


_LR_DECAY_FRACTIONS = (0.70, 0.90)


def _ensure_python312_pkg_resources_compat() -> None:
    """Shim removed pkgutil symbols for old pkg_resources consumers on Python 3.12+."""
    if not hasattr(pkgutil, "ImpImporter"):
        class ImpImporter:  # pragma: no cover - compatibility shim
            pass

        pkgutil.ImpImporter = ImpImporter
    if not hasattr(pkgutil, "ImpLoader"):
        class ImpLoader:  # pragma: no cover - compatibility shim
            pass

        pkgutil.ImpLoader = ImpLoader
    if not hasattr(FileFinder, "find_module"):
        def find_module(self, fullname: str, path: object | None = None):
            spec = self.find_spec(fullname)
            return spec.loader if spec is not None else None

        FileFinder.find_module = find_module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m object_detection.stable_dino.train",
        description="Train StableDINO with its own dataset config.",
    )
    parser.add_argument("--dataset", required=True, help="Path to Stable-DINO dataset config")
    default_config = Path(__file__).resolve().parent / "projects" / "stabledino" / "configs" / "damage_detector_stabledino_r50_4scale_12ep.py"
    parser.add_argument(
        "--config-file",
        default=str(default_config),
        help="StableDINO LazyConfig file",
    )
    parser.add_argument("--output-dir", default="object_detection/stable_dino/train")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--finetune-checkpoint", default="", help="Detector checkpoint to fine-tune with class-head mismatch filtering.")
    parser.add_argument("--finetune-ignore-prefix", nargs="*", default=["class_embed", "label_enc"], help="Checkpoint key prefixes to skip during fine-tune loading.")
    parser.add_argument("--finetune-strict", action="store_true", help="Do not skip checkpoint tensors with mismatched shapes.")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None, help="Use pinned host memory for the train dataloader.")
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=None, help="Keep train dataloader workers alive across epochs.")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="Train dataloader prefetch_factor when workers > 0.")
    parser.add_argument("--max-iter", type=int, default=None, help="Override StableDINO train.max_iter.")
    parser.add_argument("--eval-period", type=int, default=None, help="Override StableDINO train.eval_period. 0 disables periodic eval.")
    parser.add_argument("--log-period", type=int, default=None, help="Override StableDINO train.log_period.")
    parser.add_argument("--checkpoint-period", type=int, default=None, help="Override StableDINO train.checkpointer.period.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None, help="Enable CUDA automatic mixed precision training.")
    parser.add_argument(
        "--scale-lr-schedule",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Scale the 70%%/90%% LR decay milestones and warmup_length to --max-iter.",
    )
    parser.add_argument("--test-with-nms", type=float, default=0.8, help="Run validation again with this NMS threshold. Use 0 to disable.")
    parser.add_argument("--best-checkpoint-metric", default="bbox/AP", help="Metric used to save model_best.pth.")
    parser.add_argument("--best-checkpoint-mode", default="max", choices=["max", "min"], help="Best checkpoint comparison mode.")
    parser.add_argument("--no-best-checkpoint", action="store_true", help="Disable saving model_best.pth during eval.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--augmentation-profile", default="balanced", choices=["light", "balanced", "aggressive"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-split", default="val", choices=["train", "val", "test"], help="Dataset split used by eval-only and periodic evaluation.")
    parser.add_argument("--num-gpus", type=int, default=0, help="How many CUDA GPUs to launch. 0 = all available.")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0)
    parser.add_argument("--dist-url", default="auto")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Extra LazyConfig overrides")
    return parser


def _resolve_launch_num_gpus(device: str, requested_num_gpus: int) -> int:
    resolved_device = str(device or "").strip().lower()
    requested = int(requested_num_gpus or 0)
    if resolved_device != "cuda":
        return 1
    try:
        torch = get_torch()
        available = int(torch.cuda.device_count())
    except Exception:
        available = 0
    if available <= 0:
        return 1
    if requested <= 0:
        return available
    return max(1, min(requested, available))


def _build_scaled_lr_schedule_overrides(max_iter: int) -> list[str]:
    total = int(max_iter)
    if total <= 0:
        raise ValueError("--scale-lr-schedule requires --max-iter to be positive")
    milestones = sorted({max(1, min(total, int(round(total * fraction)))) for fraction in _LR_DECAY_FRACTIONS})
    warmup_iters = min(1000, max(1, total // 10))
    return [
        f"lr_multiplier.scheduler.milestones={milestones!r}",
        f"lr_multiplier.scheduler.num_updates={total}",
        f"lr_multiplier.warmup_length={warmup_iters / float(total)}",
    ]


def _worker_main(args: argparse.Namespace) -> None:
    _ensure_python312_pkg_resources_compat()
    from object_detection.stable_dino.tools import train_net

    dataset = load_stable_dino_dataset_config(args.dataset)
    resolved_device = select_device_str(args.device)
    dataset_prefix = f"stable_dino_{dataset.yaml_path.stem}"
    cache_root = Path(args.output_dir).expanduser().resolve() / "dataset_cache"
    overrides = build_stable_dino_overrides(
        dataset,
        dataset_name_prefix=dataset_prefix,
        cache_root=cache_root,
        augmentation_profile=args.augmentation_profile,
        image_size=int(args.imgsz),
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        device=resolved_device,
        output_dir=str(Path(args.output_dir).expanduser().resolve()),
        init_checkpoint=str(args.init_checkpoint or "").strip() or None,
        eval_split=str(args.eval_split),
    )
    if bool(args.scale_lr_schedule) and args.max_iter is None:
        raise ValueError("--scale-lr-schedule requires --max-iter")
    if args.max_iter is not None:
        overrides.append(f"train.max_iter={int(args.max_iter)}")
        if bool(args.scale_lr_schedule):
            overrides.extend(_build_scaled_lr_schedule_overrides(int(args.max_iter)))
    if args.eval_period is not None:
        overrides.append(f"train.eval_period={int(args.eval_period)}")
    if args.log_period is not None:
        overrides.append(f"train.log_period={int(args.log_period)}")
    if args.checkpoint_period is not None:
        overrides.append(f"train.checkpointer.period={int(args.checkpoint_period)}")
    if args.amp is not None:
        overrides.append(f"train.amp.enabled={bool(args.amp)}")
    if args.test_with_nms is not None:
        overrides.append(f"train.test_with_nms={float(args.test_with_nms)}")
    finetune_checkpoint = str(args.finetune_checkpoint or "").strip()
    if finetune_checkpoint:
        overrides.append(f"train.finetune_checkpoint.path={json.dumps(finetune_checkpoint)}")
        overrides.append(f"train.finetune_checkpoint.ignore_prefixes={json.dumps(list(args.finetune_ignore_prefix or []))}")
        overrides.append(f"train.finetune_checkpoint.ignore_shape_mismatch={not bool(args.finetune_strict)}")
    overrides.append(f"train.best_checkpointer.enabled={not bool(args.no_best_checkpoint)}")
    overrides.append(f"train.best_checkpointer.metric={args.best_checkpoint_metric!r}")
    overrides.append(f"train.best_checkpointer.mode={args.best_checkpoint_mode!r}")
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
    _ensure_python312_pkg_resources_compat()
    parser = build_parser()
    args = parser.parse_args(argv)
    finetune_checkpoint_arg = str(args.finetune_checkpoint or "").strip()
    finetune_checkpoint = Path(finetune_checkpoint_arg).expanduser() if finetune_checkpoint_arg else None
    if finetune_checkpoint is not None and not finetune_checkpoint.exists():
        raise FileNotFoundError(
            f"Fine-tune checkpoint not found: {finetune_checkpoint}. "
            "Download or copy the checkpoint before launching distributed training."
        )
    resolved_device = select_device_str(args.device)
    launch_num_gpus = _resolve_launch_num_gpus(resolved_device, int(args.num_gpus))
    args.num_gpus = int(launch_num_gpus)
    try:
        from detectron2.engine import launch
    except Exception as exc:
        raise RuntimeError(
            "StableDINO training requires Detectron2. Install a Detectron2 wheel matching "
            "your Torch/CUDA/Python environment, then rerun this command. The base "
            "requirements_dino.txt no longer installs Detectron2 from source because that "
            "often fails on Colab."
        ) from exc

    launch(
        _worker_main,
        int(launch_num_gpus),
        num_machines=int(args.num_machines),
        machine_rank=int(args.machine_rank),
        dist_url=str(args.dist_url),
        args=(args,),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
