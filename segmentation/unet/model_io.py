import json
import os

import segmentation_models_pytorch as smp

from torch_runtime import get_torch, nn


DEFAULT_MODEL_CONFIG = {
    "arch": "Unet",
    "encoder_name": "efficientnet-b4",
    "encoder_weights": None,
    "in_channels": 3,
    "classes": 1,
    "activation": None,
    "decoder_attention_type": "scse",
    "use_centerline_head": False,
}


class UnetWithCenterline(nn.Module):
    def __init__(self, base_unet):
        super().__init__()
        self.unet = base_unet
        in_channels = self._segmentation_head_in_channels(base_unet)
        self.centerline_head = nn.Conv2d(in_channels, 1, kernel_size=1)

    @staticmethod
    def _segmentation_head_in_channels(base_unet):
        for module in base_unet.segmentation_head.modules():
            if isinstance(module, nn.Conv2d):
                return int(module.in_channels)
        raise RuntimeError("Could not infer UNet decoder output channels for centerline head.")

    def forward(self, x):
        features = self.unet.encoder(x)
        try:
            decoder_output = self.unet.decoder(features)
        except TypeError:
            decoder_output = self.unet.decoder(*features)
        mask_logits = self.unet.segmentation_head(decoder_output)
        centerline_logits = self.centerline_head(decoder_output)
        return mask_logits, centerline_logits


def build_model_config(args):
    config = dict(DEFAULT_MODEL_CONFIG)
    config["encoder_name"] = getattr(args, "encoder_name", config["encoder_name"])
    config["encoder_weights"] = getattr(args, "encoder_weights", config["encoder_weights"])
    config["use_centerline_head"] = bool(
        getattr(args, "use_centerline_head", False) or float(getattr(args, "centerline_weight", 0.0) or 0.0) > 0.0
    )
    return config


def _state_dict(model):
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    return state_dict


def _checkpoint_payload(model, model_config, epoch=None, metrics=None, ema_model=None, ema_decay=None):
    state_dict = _state_dict(model)
    payload = {
        "state_dict": state_dict,
        "model_config": dict(model_config or DEFAULT_MODEL_CONFIG),
    }
    if ema_model is not None:
        payload["ema_state_dict"] = _state_dict(ema_model)
        if ema_decay is not None:
            payload["ema_decay"] = float(ema_decay)
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if metrics:
        payload["metrics"] = dict(metrics)
    return payload


def save_checkpoint(path, model, model_config, epoch=None, metrics=None, ema_model=None, ema_decay=None):
    torch = get_torch()
    torch.save(
        _checkpoint_payload(
            model,
            model_config,
            epoch=epoch,
            metrics=metrics,
            ema_model=ema_model,
            ema_decay=ema_decay,
        ),
        path,
    )


def extract_state_dict_and_config(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model_config = checkpoint.get("model_config") or {}
        return state_dict, model_config
    return checkpoint, {}


def _build_model(model_config, *, encoder_weights=None):
    base_unet = smp.Unet(
        encoder_name=model_config["encoder_name"],
        encoder_weights=encoder_weights,
        in_channels=int(model_config.get("in_channels", 3)),
        classes=int(model_config.get("classes", 1)),
        activation=model_config.get("activation"),
        decoder_attention_type=model_config.get("decoder_attention_type"),
    )
    if bool(model_config.get("use_centerline_head", False)):
        return UnetWithCenterline(base_unet)
    return base_unet


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
    model = _build_model(model_config, encoder_weights=None)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict, _ = extract_state_dict_and_config(checkpoint)
    if isinstance(checkpoint, dict) and checkpoint.get("ema_state_dict") is not None:
        state_dict = checkpoint["ema_state_dict"]
    model.load_state_dict(state_dict, strict=strict)
    model = model.to(device)
    model.eval()
    return model, model_config
