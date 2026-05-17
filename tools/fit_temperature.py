"""Fit scalar temperature scaling for binary segmentation logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_tensor(path: str | Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return payload.float()
    if isinstance(payload, dict):
        for key in ("logits", "targets", "labels", "masks", "tensor"):
            value = payload.get(key)
            if isinstance(value, torch.Tensor):
                return value.float()
    raise TypeError(f"Unsupported tensor payload: {path}")


def _load_from_dir(logits_dir: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    root = Path(logits_dir)
    candidates = [
        (root / "val_logits.pt", root / "val_targets.pt"),
        (root / "logits.pt", root / "targets.pt"),
    ]
    for logits_path, targets_path in candidates:
        if logits_path.is_file() and targets_path.is_file():
            return _load_tensor(logits_path), _load_tensor(targets_path)

    pairs = []
    for logits_path in sorted(root.glob("*logits*.pt")):
        target_name = logits_path.name.replace("logits", "targets")
        target_path = logits_path.with_name(target_name)
        if target_path.is_file():
            pairs.append((logits_path, target_path))
    if not pairs:
        raise FileNotFoundError(f"No logits/targets .pt pairs found in {root}")
    logits = [_load_tensor(path) for path, _target in pairs]
    targets = [_load_tensor(path) for _logits, path in pairs]
    return torch.cat([x.reshape(-1) for x in logits]), torch.cat([x.reshape(-1) for x in targets])


def fit_temperature(
    val_logits: torch.Tensor,
    val_targets: torch.Tensor,
    *,
    init_temperature: float = 1.0,
    lr: float = 0.01,
    max_iter: int = 200,
) -> float:
    logits = val_logits.float().reshape(-1)
    targets = (val_targets.float().reshape(-1) > 0.5).float()
    if logits.numel() != targets.numel():
        raise ValueError(f"logits/targets size mismatch: {logits.numel()} vs {targets.numel()}")

    log_t = torch.tensor([float(init_temperature)], dtype=torch.float32).clamp_min(1e-4).log().requires_grad_(True)
    optimizer = torch.optim.LBFGS([log_t], lr=float(lr), max_iter=int(max_iter), line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = log_t.exp().clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, targets)
        loss.backward()
        return loss

    before = float(F.binary_cross_entropy_with_logits(logits, targets).item())
    optimizer.step(closure)
    temperature = float(log_t.detach().exp().clamp(0.05, 20.0).item())
    after = float(F.binary_cross_entropy_with_logits(logits / temperature, targets).item())
    print(f"temperature={temperature:.6f} bce_before={before:.6f} bce_after={after:.6f}")
    return temperature


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits", default="", help="Path to logits tensor .pt")
    parser.add_argument("--targets", default="", help="Path to target tensor .pt")
    parser.add_argument("--logits_dir", default="", help="Directory containing logits/targets .pt files")
    parser.add_argument("--output", default="", help="Output JSON path; defaults to best_temperature.json next to inputs")
    parser.add_argument("--init_temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max_iter", type=int, default=200)
    args = parser.parse_args()

    if args.logits and args.targets:
        logits = _load_tensor(args.logits)
        targets = _load_tensor(args.targets)
        default_output_dir = Path(args.logits).resolve().parent
    elif args.logits_dir:
        logits, targets = _load_from_dir(args.logits_dir)
        default_output_dir = Path(args.logits_dir).resolve()
    else:
        raise ValueError("Pass either --logits and --targets, or --logits_dir.")

    temperature = fit_temperature(
        logits,
        targets,
        init_temperature=float(args.init_temperature),
        lr=float(args.lr),
        max_iter=int(args.max_iter),
    )
    output_path = Path(args.output) if args.output else default_output_dir / "best_temperature.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"temperature": float(temperature)}, f, indent=2, sort_keys=True)
    print(f"saved={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
