"""Detection-only dataset mapper.

Fixes a bug in older detrex COCOInstanceNewBaselineDatasetMapper where
`instances.gt_masks = PolygonMasks([])` is called unconditionally, crashing
on bbox-only datasets (no polygon segmentation annotations).

Uses the same coco_instance_transform_gen augmentation interface so
build_stable_dino_overrides overrides (augmentation.image_size etc.) still work.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detrex.data.dataset_mappers import COCOInstanceNewBaselineDatasetMapper


class DetectionDatasetMapper(COCOInstanceNewBaselineDatasetMapper):
    """Drop-in replacement that skips gt_masks for bbox-only datasets."""

    def __call__(self, dataset_dict: dict) -> dict:
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        padding_mask = np.ones(image.shape[:2])
        image, transforms = T.apply_transform_gens(self.augmentation, image)
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~padding_mask.astype(bool)
        image_shape = image.shape[:2]

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            if instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances = utils.filter_empty_instances(instances)
            dataset_dict["instances"] = instances

        return dataset_dict
