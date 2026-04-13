from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from data_split.types import SampleRecord


def group_embeddings(records: list[SampleRecord], embeddings: np.ndarray) -> pd.DataFrame:
    grouped_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    grouped_records: dict[str, list[SampleRecord]] = defaultdict(list)
    for record, embedding in zip(records, embeddings):
        grouped_vectors[record.source_id].append(embedding)
        grouped_records[record.source_id].append(record)

    rows: list[dict[str, object]] = []
    for source_id in sorted(grouped_vectors):
        source_vectors = np.stack(grouped_vectors[source_id], axis=0)
        source_records = grouped_records[source_id]
        rows.append(
            {
                "source_id": source_id,
                "image_count": len(source_records),
                "mask_positive_sum": float(sum(record.positive_ratio for record in source_records)),
                "mask_positive_mean": float(np.mean([record.positive_ratio for record in source_records])),
                "embedding": source_vectors.mean(axis=0).astype(np.float32),
            }
        )
    return pd.DataFrame(rows)


def cluster_sources(group_df: pd.DataFrame, requested_clusters: int) -> pd.DataFrame:
    num_groups = int(group_df.shape[0])
    if num_groups < 3:
        raise ValueError("At least three source groups are required to create train/val/test.")

    n_clusters = min(num_groups, max(3, int(requested_clusters)))
    matrix = np.stack(group_df["embedding"].to_list(), axis=0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-12, None)

    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=min(1024, max(128, n_clusters * 16)),
        n_init=10,
    )
    labels = model.fit_predict(matrix)
    output = group_df.copy()
    unique_labels = np.unique(labels)
    if unique_labels.size < min(3, num_groups):
        output["cluster_id"] = np.arange(num_groups, dtype=int)
    else:
        output["cluster_id"] = labels.astype(int)
    return output


def assign_sources(group_df: pd.DataFrame, split_names: list[str], split_ratios: list[float]) -> tuple[dict[str, str], pd.DataFrame]:
    source_df = group_df.sort_values(["image_count", "mask_positive_mean", "source_id"], ascending=[False, False, True]).reset_index(drop=True)

    total_images = float(source_df["image_count"].sum())
    total_mask = float(source_df["mask_positive_sum"].sum())
    image_targets = {name: ratio * total_images for name, ratio in zip(split_names, split_ratios)}
    mask_targets = {name: ratio * total_mask for name, ratio in zip(split_names, split_ratios)}
    image_state = {name: 0.0 for name in split_names}
    mask_state = {name: 0.0 for name in split_names}
    source_state = {name: 0 for name in split_names}
    cluster_totals = group_df.groupby("cluster_id")["image_count"].sum().to_dict()
    cluster_split_state: dict[int, dict[str, float]] = {
        int(cluster_id): {split_name: 0.0 for split_name in split_names}
        for cluster_id in group_df["cluster_id"].unique()
    }

    source_to_split: dict[str, str] = {}
    remaining_rows = list(source_df.itertuples(index=False))
    if len(remaining_rows) >= len(split_names):
        seeded = remaining_rows[:len(split_names)]
        remaining_rows = remaining_rows[len(split_names):]
        for split_name, row in zip(split_names, seeded):
            source_to_split[str(row.source_id)] = split_name
            image_state[split_name] += float(row.image_count)
            mask_state[split_name] += float(row.mask_positive_sum)
            source_state[split_name] += 1
            cluster_split_state[int(row.cluster_id)][split_name] += float(row.image_count)

    for row in remaining_rows:
        best_split = split_names[0]
        best_cost = math.inf
        for split_name in split_names:
            next_images = image_state.copy()
            next_masks = mask_state.copy()
            next_sources = source_state.copy()
            next_images[split_name] += float(row.image_count)
            next_masks[split_name] += float(row.mask_positive_sum)
            next_sources[split_name] += 1

            size_cost = sum(
                ((next_images[name] - image_targets[name]) / max(image_targets[name], 1.0)) ** 2
                for name in split_names
            )
            if total_mask > 0:
                mask_cost = sum(
                    ((next_masks[name] - mask_targets[name]) / max(mask_targets[name], 1e-6)) ** 2
                    for name in split_names
                )
            else:
                mask_cost = 0.0
            overflow = max(0.0, next_images[split_name] - image_targets[split_name] * 1.03) / max(image_targets[split_name], 1.0)
            source_balance = sum(
                ((next_sources[name] - source_df.shape[0] * split_ratios[idx]) / max(source_df.shape[0] * split_ratios[idx], 1.0)) ** 2
                for idx, name in enumerate(split_names)
            )

            cluster_state = cluster_split_state[int(row.cluster_id)]
            occupied_splits = [name for name, value in cluster_state.items() if value > 0]
            cluster_penalty = 0.0
            if occupied_splits:
                majority_split = max(cluster_state, key=cluster_state.get)
                if split_name not in occupied_splits:
                    cluster_penalty += 0.8 * len(occupied_splits)
                if split_name != majority_split:
                    cluster_penalty += cluster_state[majority_split] / max(float(cluster_totals[int(row.cluster_id)]), 1.0)

            cost = size_cost + 0.15 * mask_cost + 25.0 * (overflow ** 2) + 0.02 * source_balance + 0.05 * cluster_penalty
            if cost < best_cost:
                best_cost = cost
                best_split = split_name

        source_to_split[str(row.source_id)] = best_split
        image_state[best_split] += float(row.image_count)
        mask_state[best_split] += float(row.mask_positive_sum)
        source_state[best_split] += 1
        cluster_split_state[int(row.cluster_id)][best_split] += float(row.image_count)

    summary_rows = []
    for split_name, ratio in zip(split_names, split_ratios):
        split_frame = group_df[group_df["source_id"].map(source_to_split.get) == split_name]
        image_count = int(split_frame["image_count"].sum())
        source_count = int(split_frame.shape[0])
        cluster_count = int(split_frame["cluster_id"].nunique())
        mask_sum = float(split_frame["mask_positive_sum"].sum())
        summary_rows.append(
            {
                "split": split_name,
                "target_ratio": ratio,
                "actual_ratio": image_count / total_images if total_images else 0.0,
                "image_count": image_count,
                "source_count": source_count,
                "cluster_count": cluster_count,
                "mask_positive_mean": mask_sum / image_count if image_count else 0.0,
            }
        )
    summary_rows.append(
        {
            "split": "overall",
            "target_ratio": 1.0,
            "actual_ratio": 1.0,
            "image_count": int(total_images),
            "source_count": int(group_df.shape[0]),
            "cluster_count": int(group_df["cluster_id"].nunique()),
            "mask_positive_mean": total_mask / total_images if total_images else 0.0,
        }
    )
    return source_to_split, pd.DataFrame(summary_rows)


def build_assignment_table(records: list[SampleRecord], group_df: pd.DataFrame, source_to_split: dict[str, str], input_root: Path) -> pd.DataFrame:
    source_to_cluster = dict(zip(group_df["source_id"], group_df["cluster_id"]))
    cluster_sizes = group_df.groupby("cluster_id")["image_count"].sum().to_dict()
    source_sizes = group_df.set_index("source_id")["image_count"].to_dict()

    rows = []
    images_dir = input_root / "images"
    masks_dir = input_root / "masks"
    for record in records:
        cluster_id = int(source_to_cluster[record.source_id])
        split_name = source_to_split[record.source_id]
        relative_image = record.image_path.relative_to(images_dir)
        relative_mask = record.mask_path.relative_to(masks_dir) if record.mask_path is not None and masks_dir.is_dir() else None
        rows.append(
            {
                "filename": record.image_path.name,
                "stem": record.stem,
                "split": split_name,
                "sample_weight": 1.0,
                "source_id": record.source_id,
                "source_weight": int(source_sizes[record.source_id]),
                "cluster_id": cluster_id,
                "cluster_weight": int(cluster_sizes[cluster_id]),
                "mask_positive_ratio": float(record.positive_ratio),
                "image_path": str(record.image_path),
                "mask_path": str(record.mask_path) if record.mask_path is not None else "",
                "relative_image_path": str(relative_image),
                "relative_mask_path": str(relative_mask) if relative_mask is not None else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "source_id", "filename"]).reset_index(drop=True)
