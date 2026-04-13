from __future__ import annotations

from data_split.cli import parse_args
from data_split.config import SplitConfig
from data_split.runner import run_split


def main() -> None:
    args = parse_args()
    config = SplitConfig(
        input_root=args.input_root,
        output_root=args.output_root,
        splits=tuple(args.splits),
        split_names=tuple(args.split_names),
        num_clusters=args.num_clusters,
        batch_size=args.batch_size,
        device=args.device,
        checkpoint=args.checkpoint,
        mask_threshold=args.mask_threshold,
    )
    run_split(config)


if __name__ == "__main__":
    main()
