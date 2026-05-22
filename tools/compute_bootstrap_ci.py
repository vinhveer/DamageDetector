"""Bootstrap 95% CI for per-image metrics (S9.2).

Reads per_image_metrics.csv from each (model, dataset), computes percentile bootstrap CI.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_LABEL_MAP = {
    "crack500": "crack500_test",
    "volker": "crack_segmentation_dataset_volker_test",
    "deepcrack": "deepcrack_test",
}


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000,
                 confidence: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - confidence) / 2
    lo = float(np.percentile(boot_means, alpha * 100))
    hi = float(np.percentile(boot_means, (1 - alpha) * 100))
    return float(values.mean()), lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--metrics", nargs="+", default=["dice", "iou", "skeleton_dice"])
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    rows = []

    for model in args.models:
        for dataset in args.datasets:
            ds_label = DATASET_LABEL_MAP[dataset]
            csv_path = eval_root / model / ds_label / "per_image_metrics.csv"
            if not csv_path.exists():
                print(f"SKIP: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            for metric in args.metrics:
                if metric not in df.columns:
                    continue
                values = df[metric].dropna().to_numpy()
                n = len(values)
                if n == 0:
                    continue
                if n < 30:
                    print(f"WARN: {model}/{dataset}/{metric}: n={n} < 30")

                mean, lo, hi = bootstrap_ci(
                    values, n_boot=args.n_bootstrap,
                    confidence=args.confidence_level, seed=args.random_seed,
                )
                rows.append({
                    "model": model, "dataset": dataset, "metric": metric,
                    "n": n, "mean": mean, "ci_lo": lo, "ci_hi": hi,
                    "ci_width": hi - lo,
                })

    out_df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
