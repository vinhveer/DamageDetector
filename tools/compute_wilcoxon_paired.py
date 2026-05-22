"""Wilcoxon signed-rank paired comparison (S9.3).

Reads per_image_metrics.csv, pairs by image_id, computes Wilcoxon + effect size.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DATASET_LABEL_MAP = {
    "crack500": "crack500_test",
    "volker": "crack_segmentation_dataset_volker_test",
    "deepcrack": "deepcrack_test",
}


def _load_per_image(eval_root: Path, model: str, dataset: str) -> pd.DataFrame:
    ds_label = DATASET_LABEL_MAP[dataset]
    csv_path = eval_root / model / ds_label / "per_image_metrics.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df["image_id"] = df["image_id"].str.lower().str.strip()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--metrics", nargs="+", default=["dice", "iou", "skeleton_dice"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    pairs_df = pd.read_csv(args.pairs_csv)
    rows = []

    for _, pair in pairs_df.iterrows():
        model_a, model_b, dataset = pair["model_a"], pair["model_b"], pair["dataset"]
        df_a = _load_per_image(eval_root, model_a, dataset)
        df_b = _load_per_image(eval_root, model_b, dataset)
        if df_a.empty or df_b.empty:
            print(f"SKIP: {model_a} vs {model_b} on {dataset} (missing data)")
            continue

        merged = df_a.merge(df_b, on="image_id", suffixes=("_a", "_b"))
        n_paired = len(merged)
        if n_paired < 10:
            for metric in args.metrics:
                rows.append({
                    "model_a": model_a, "model_b": model_b, "dataset": dataset,
                    "metric": metric, "n_paired": n_paired,
                    "median_diff": float("nan"), "wilcoxon_stat": float("nan"),
                    "p_value": float("nan"), "significant": False,
                    "effect_size_rb": float("nan"), "note": "n_paired < 10",
                })
            continue

        for metric in args.metrics:
            col_a = f"{metric}_a"
            col_b = f"{metric}_b"
            if col_a not in merged.columns or col_b not in merged.columns:
                continue

            diff = merged[col_a].to_numpy() - merged[col_b].to_numpy()
            # Drop NaN
            valid = ~np.isnan(diff)
            diff = diff[valid]
            n = len(diff)
            if n < 10:
                rows.append({
                    "model_a": model_a, "model_b": model_b, "dataset": dataset,
                    "metric": metric, "n_paired": n,
                    "median_diff": float("nan"), "wilcoxon_stat": float("nan"),
                    "p_value": float("nan"), "significant": False,
                    "effect_size_rb": float("nan"), "note": "n < 10 after NaN drop",
                })
                continue

            # Check if all zeros
            if np.all(diff == 0):
                rows.append({
                    "model_a": model_a, "model_b": model_b, "dataset": dataset,
                    "metric": metric, "n_paired": n,
                    "median_diff": 0.0, "wilcoxon_stat": 0.0,
                    "p_value": 1.0, "significant": False,
                    "effect_size_rb": 0.0, "note": "identical predictions",
                })
                continue

            stat, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            median_diff = float(np.median(diff))
            # Rank-biserial correlation
            r_rb = 1.0 - (2.0 * stat) / (n * (n + 1) / 2.0)

            rows.append({
                "model_a": model_a, "model_b": model_b, "dataset": dataset,
                "metric": metric, "n_paired": n,
                "median_diff": median_diff, "wilcoxon_stat": float(stat),
                "p_value": float(p), "significant": p < args.alpha,
                "effect_size_rb": float(r_rb), "note": "",
            })

    out_df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
