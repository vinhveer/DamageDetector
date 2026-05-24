"""Detection-only dataloader config (no polygon masks).

Replaces coco_instance_seg.py which uses COCOInstanceNewBaselineDatasetMapper —
an instance-segmentation mapper that crashes on bbox-only datasets.

Uses DetectionDatasetMapper (same coco_instance_transform_gen augmentation
interface) so build_stable_dino_overrides augmentation overrides still work.
"""

from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator

from detrex.data.dataset_mappers import coco_instance_transform_gen

from .detection_mapper import DetectionDatasetMapper

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_2017_train"),
    mapper=L(DetectionDatasetMapper)(
        augmentation=L(coco_instance_transform_gen)(
            image_size=1024,
            min_scale=0.1,
            max_scale=2.0,
            random_flip="horizontal",
        ),
        is_train=True,
        image_format="RGB",
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_2017_val", filter_empty=False),
    mapper=L(DetectionDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        is_train=False,
        image_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
