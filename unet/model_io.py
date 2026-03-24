import json
import os

import segmentation_models_pytorch as smp

from torch_runtime import get_torch


DEFAULT_MODEL_CONFIG = {
    "arch": "Unet",
    "encoder_name": "efficientnet-b4",
    "encoder_weights": None,
    "in_channels": 3,
    "classes": 1,
    "activation": None,
    "decoder_attention_type": "scse",
}


def build_model_config(args):
    config = dict(DEFAULT_MODEL_CONFIG)
    config["encoder_name"] = getattr(args, "encoder_name", config["encoder_name"])
    config["encoder_weights"] = getattr(args, "encoder_weights", config["encoder_weights"])
    return config


def _checkpoint_payload(model, model_config, epoch=None, metrics=None):
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    payload = {
        "state_dict": state_dict,
        "model_config": dict(model_config or DEFAULT_MODEL_CONFIG),
    }
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if metrics:
        payload["metrics"] = dict(metrics)
    return payload


def save_checkpoint(path, model, model_config, epoch=None, metrics=None):
    torch = get_torch()
    torch.save(_checkpoint_payload(model, model_config, epoch=epoch, metrics=metrics), path)


def extract_state_dict_and_config(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model_config = checkpoint.get("model_config") or {}
        return state_dict, model_config
    return checkpoint, {}


def _sidecar_config_path(model_path):
    return os.path.join(os.path.dirname(os.path.abspath(model_path)), "train_config.json")


def save_training_config(output_dir, args, *, model_config=None, train_preprocess=None, val_preprocess=None):
    payload = {
        "args": dict(vars(args)),
        "model_config": dict(model_config or DEFAULT_MODEL_CONFIG),
        "resolved": {
            "train_preprocess": train_preprocess,
            "val_preprocess": val_preprocess,
        },
    }
    config_path = os.path.join(output_dir, "train_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return config_path


def load_model_config_from_path(model_path):
    torch = get_torch()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    _, checkpoint_config = extract_state_dict_and_config(checkpoint)
    merged = dict(DEFAULT_MODEL_CONFIG)
    merged.update(checkpoint_config or {})

    sidecar_path = _sidecar_config_path(model_path)
    if os.path.exists(sidecar_path):
        with open(sidecar_path, "r", encoding="utf-8") as f:
            sidecar = json.load(f)
        sidecar_config = sidecar.get("model_config") or {}
        merged.update(sidecar_config)

    return merged


def load_training_config_from_path(model_path):
    sidecar_path = _sidecar_config_path(model_path)
    if not os.path.exists(sidecar_path):
        return None
    with open(sidecar_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_from_checkpoint(model_path, device, *, strict=True):
    torch = get_torch()
    model_config = load_model_config_from_path(model_path)
    model = smp.Unet(
        encoder_name=model_config["encoder_name"],
        encoder_weights=None,
        in_channels=int(model_config.get("in_channels", 3)),
        classes=int(model_config.get("classes", 1)),
        activation=model_config.get("activation"),
        decoder_attention_type=model_config.get("decoder_attention_type"),
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict, _ = extract_state_dict_and_config(checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    model = model.to(device)
    model.eval()
    return model, model_config
