from __future__ import annotations

from pathlib import Path

from data_split.config import SplitConfig
from data_split.runner import run_split


def main() -> None:
    config = SplitConfig(
        input_root=Path("/Users/nguyenquangvinh/Desktop/Lab/OldWork/RawDatasets/crack500"),
        output_root=Path("/Users/nguyenquangvinh/Desktop/Lab/data/datasets/crack500"),
        splits=(0.7, 0.15, 0.15),
        split_names=("train", "val", "test"),
        num_clusters=32,
        batch_size=16,
        device="auto",
        mask_threshold=127,
    )
    run_split(config)


if __name__ == "__main__":
    main()
