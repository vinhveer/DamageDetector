from __future__ import annotations

import argparse
from pathlib import Path

from object_detection.dinov2.dinov2_prototypes import default_dinov2_embedding_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split an image dataset into train/val/test using DINOv2 embeddings and DataSAIL-style clustering.",
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Input dataset root. Expected images/ and optional masks/ folders.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root for workbook, cache, and exported splits.")
    parser.add_argument("--splits", nargs=3, type=float, default=[0.7, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--split-names", nargs=3, default=["train", "val", "test"], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--num-clusters", type=int, default=36, help="Number of visual clusters across source groups.")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Torch device preference.")
    parser.add_argument("--checkpoint", default=default_dinov2_embedding_checkpoint(), help="Local DINOv2 checkpoint folder or model id.")
    parser.add_argument("--mask-threshold", type=int, default=127, help="Threshold for binary mask coverage estimation.")
    return parser.parse_args()
