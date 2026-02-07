import argparse
import os

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train U-Net for crack segmentation.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config file path.",
    )
    return parser


DEFAULT_CONFIG = {
    "train_images": "",
    "train_masks": "",
    "val_images": "",
    "val_masks": "",
    "mask_prefix": "auto",
    "output_dir": "output_results",
    "preprocess": "patch",
    "input_size": 512,
    "patches_per_image": 2,
    "max_patch_tries": 5,
    "val_stride": 0,
    "batch_size": 16,
    "num_workers": 8,
    "epochs": 80,
    "seed": 42,
    "no_augment": False,
    "cache_dir": None,
    "cache_rebuild": False,
    "no_visualize": True,
    "no_loss_curve": True,
    "pos_weight": 5.0,
    "bce_weight": 0.4,
    "dice_weight": 0.6,
    "metric_threshold": 0.5,
    "metric_thresholds": "",
    "scheduler_metric": "loss",
    "learning_rate": 0.0005,
    "weight_decay": 0.00001,
    "scheduler_factor": 0.5,
    "scheduler_patience": 10,
    "early_stop_patience": 15,
    "visualize_every": 0,
    "prefetch_factor": 2,
    "persistent_workers": True,
    "pin_memory": None,
    "grad_accum_steps": 1,
    "encoder_name": "efficientnet-b4",
    "encoder_weights": "imagenet",
    "scheduler_t0": 10,
    "scheduler_tmult": 2,
}

REQUIRED_KEYS = ["train_images", "train_masks", "val_images", "val_masks"]


def _coerce_config_value(key, value):
    if key == "metric_thresholds" and isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return value


def load_config(config_path):
    if not config_path:
        raise RuntimeError("Missing --config path.")
    if not os.path.exists(config_path):
        raise RuntimeError(f"Config file not found: {config_path}")
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to use --config. Install with: pip install PyYAML"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError("Config file must be a YAML mapping (key: value).")

    merged = dict(DEFAULT_CONFIG)
    for key, value in data.items():
        if key not in merged:
            raise RuntimeError(f"Unknown config key: {key}")
        merged[key] = _coerce_config_value(key, value)
    merged["config"] = config_path
    merged["_config_used"] = True
    return argparse.Namespace(**merged)


def validate_args(args):
    missing = [name for name in REQUIRED_KEYS if not getattr(args, name, None)]
    if missing:
        raise RuntimeError(
            "Missing required values: " + ", ".join(missing) + ". "
            "Provide them in the config file."
        )
