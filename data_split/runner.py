from __future__ import annotations

from data_split.assign import assign_sources, build_assignment_table, cluster_sources, group_embeddings
from data_split.config import SplitConfig
from data_split.dataset import discover_samples, normalize_split_ratios
from data_split.embedding import embed_images
from data_split.export import export_split_folders, write_workbook


def run_split(config: SplitConfig) -> None:
    input_root = config.input_root.expanduser().resolve()
    output_root = config.output_root.expanduser().resolve()
    split_names = list(config.split_names)
    split_ratios = normalize_split_ratios(config.splits)
    if split_names != ["train", "val", "test"]:
        raise ValueError("This exporter currently expects split names train val test to keep workbook/export layout stable.")

    print(f"Input root:  {input_root}")
    print(f"Output root: {output_root}")
    records = discover_samples(input_root, config.mask_threshold)
    print(f"Discovered {len(records)} image samples.")

    image_paths = [record.image_path for record in records]
    embeddings = embed_images(
        image_paths=image_paths,
        checkpoint=config.checkpoint,
        batch_size=config.batch_size,
        device_preference=config.device,
        cache_path=output_root / "embedding_cache.pkl",
    )
    print(f"Embedding matrix shape: {tuple(embeddings.shape)}")

    group_df = group_embeddings(records, embeddings)
    print(f"Grouped into {group_df.shape[0]} source groups.")

    clustered_df = cluster_sources(group_df, config.num_clusters)
    source_to_split, summary_df = assign_sources(clustered_df, split_names, split_ratios)
    assignments = build_assignment_table(records, clustered_df, source_to_split, input_root)

    workbook_path = write_workbook(output_root, assignments, summary_df)
    export_split_folders(output_root, assignments)

    print(f"Workbook written to: {workbook_path}")
    print(summary_df.to_string(index=False))
