"""Rebuild ntt YOLO segmentation with UNet cracks (YOLO + UNet + SAM).

Source = damage_segmentation_yolo (YOLO detector + SAM masks).
Crack masks are re-segmented with UNet; non-crack (mold/spall) masks are
copied from the YOLO/SAM output. Mirrors the gdino+unet rebuild.
"""
from pathlib import Path
from pineline.house_cutout.rebuild_gdino_with_unet_cracks import rebuild_case, RESULTS

ntt_clean = RESULTS / "nha_truyen_thong" / "damage_segmentation_clean_overlays" / "clean_house_bg.png"

rebuild_case(
    name="ntt_yolo_sam_vith_unet_crack",
    image_path=ntt_clean,
    src_dir=RESULTS / "nha_truyen_thong" / "damage_segmentation_yolo",
    out_dir=RESULTS / "nha_truyen_thong" / "damage_segmentation_yolo_sam_vith_unet_crack",
    crop_box=None,
    background=(255, 255, 255),
)
print("REBUILD DONE")
