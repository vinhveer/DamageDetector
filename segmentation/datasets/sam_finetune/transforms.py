import cv2
import numpy as np
from torch_runtime import torch

from ..core import (
    build_crack_profile_augment,
    choose_centered_foreground_crop_coords,
    choose_refine_crop_coords,
    choose_smart_crack_crop_coords,
    crop_image_label,
    ensure_numpy_image_label,
    ensure_three_channel_image,
    normalize_binary_mask,
    pad_canvas_if_needed,
    resize_with_center_padding,
    to_uint8_image,
)


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class RandomGenerator(object):
    def __init__(
        self,
        output_size,
        low_res,
        background_crop_prob: float = 0.2,
        near_background_crop_prob: float = 0.15,
        hard_negative_crop_prob: float = 0.10,
        augment_profile: str = "balanced",
        crop_policy: str = "smart",
    ):
        self.output_size = output_size
        self.low_res = low_res
        self.background_crop_prob = float(background_crop_prob)
        self.near_background_crop_prob = float(near_background_crop_prob)
        self.hard_negative_crop_prob = float(hard_negative_crop_prob)
        self.augment_profile = str(augment_profile or "balanced").strip().lower()
        self.crop_policy = str(crop_policy or "smart").strip().lower()
        self.transform = build_crack_profile_augment(self.augment_profile)

    def _choose_crop_coords(
        self,
        image: np.ndarray,
        label: np.ndarray,
        h: int,
        w: int,
        th: int,
        tw: int,
        *,
        crop_metadata: dict | None = None,
    ) -> tuple[int, int]:
        return choose_smart_crack_crop_coords(
            image,
            label,
            h,
            w,
            th,
            tw,
            background_crop_prob=self.background_crop_prob,
            near_background_crop_prob=self.near_background_crop_prob,
            hard_negative_crop_prob=self.hard_negative_crop_prob,
            crop_policy=self.crop_policy,
            metadata=crop_metadata,
        )

    def __call__(self, sample):
        image, label = ensure_numpy_image_label(sample["image"], sample["label"])
        crop_metadata = sample.get("crop_metadata")
        if image.shape[:2] != label.shape[:2]:
            raise ValueError(
                f"Image/mask size mismatch: image={image.shape[:2]} mask={label.shape[:2]}. "
                "Fix the dataset instead of silently cropping/padding labels."
            )

        image = ensure_three_channel_image(image)
        label = normalize_binary_mask(label)

        h, w, _c = image.shape
        th, tw = self.output_size[1], self.output_size[0]
        image, label = pad_canvas_if_needed(
            image,
            label,
            min_h=th,
            min_w=tw,
            image_border_mode=cv2.BORDER_REFLECT_101,
            label_border_mode=cv2.BORDER_CONSTANT,
            label_fill=0,
        )
        h, w = label.shape[:2]

        y1, x1 = self._choose_crop_coords(image, label, h, w, th, tw, crop_metadata=crop_metadata)
        image, label = crop_image_label(image, label, y1=y1, x1=x1, th=th, tw=tw)
        augmented = self.transform(image=to_uint8_image(image), mask=(label > 0).astype(np.uint8))
        final_image, final_label = resize_with_center_padding(
            augmented["image"].astype(np.float32),
            augmented["mask"].astype(np.float32),
            target_h=self.output_size[0],
            target_w=self.output_size[1],
            normalize_image=True,
        )
        return {
            "image": torch.from_numpy(final_image),
            "label": torch.from_numpy(final_label) > 0.5,
        }


class RefineRandomGenerator(RandomGenerator):
    def __init__(
        self,
        output_size,
        low_res,
        background_crop_prob: float = 0.1,
        near_background_crop_prob: float = 0.2,
        hard_negative_crop_prob: float = 0.1,
        augment_profile: str = "balanced",
        crop_policy: str = "smart",
        roi_positive_band_low: float = 0.20,
        roi_positive_band_high: float = 0.90,
    ):
        super().__init__(
            output_size=output_size,
            low_res=low_res,
            background_crop_prob=background_crop_prob,
            near_background_crop_prob=near_background_crop_prob,
            hard_negative_crop_prob=hard_negative_crop_prob,
            augment_profile=augment_profile,
            crop_policy=crop_policy,
        )
        self.roi_positive_band_low = float(roi_positive_band_low)
        self.roi_positive_band_high = float(roi_positive_band_high)

    def _choose_crop_coords(
        self,
        image: np.ndarray,
        label: np.ndarray,
        h: int,
        w: int,
        th: int,
        tw: int,
        *,
        crop_metadata: dict | None = None,
    ) -> tuple[int, int]:
        return choose_refine_crop_coords(image, label, h, w, th, tw, crop_policy=self.crop_policy, metadata=crop_metadata)


class ValGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = ensure_numpy_image_label(sample["image"], sample["label"])
        crop_metadata = sample.get("crop_metadata")
        if image.shape[:2] != label.shape[:2]:
            raise ValueError(
                f"Image/mask size mismatch: image={image.shape[:2]} mask={label.shape[:2]}. "
                "Fix the dataset instead of silently cropping/padding labels."
            )

        image = ensure_three_channel_image(image)
        label = normalize_binary_mask(label)

        h, w, _c = image.shape
        th, tw = self.output_size[1], self.output_size[0]
        image, label = pad_canvas_if_needed(
            image,
            label,
            min_h=th,
            min_w=tw,
            image_border_mode=cv2.BORDER_CONSTANT,
            image_fill=0,
            label_border_mode=cv2.BORDER_CONSTANT,
            label_fill=0,
        )
        h, w = label.shape[:2]
        y1, x1 = choose_centered_foreground_crop_coords(label, h, w, th, tw, metadata=crop_metadata)
        image, label = crop_image_label(image, label, y1=y1, x1=x1, th=th, tw=tw)
        final_image, final_label = resize_with_center_padding(
            image.astype(np.float32),
            label.astype(np.float32),
            target_h=self.output_size[0],
            target_w=self.output_size[1],
            normalize_image=True,
        )
        return {
            "image": torch.from_numpy(final_image),
            "label": torch.from_numpy(final_label) > 0.5,
        }
