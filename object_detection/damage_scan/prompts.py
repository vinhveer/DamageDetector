from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSpec:
    key: str
    prompt: str
    box_threshold: float
    text_threshold: float
    tile_bias: str
    max_side_ratio: float
    max_area_ratio: float
    min_component_area: int
    nms_iou: float
    fuse_iou: float


PROMPT_SPECS: dict[str, PromptSpec] = {
    "crack": PromptSpec(
        key="crack",
        prompt=(
            "crack, wall crack, concrete crack, thin crack, long crack, "
            "hairline crack, irregular crack, fracture line, dark crack line"
        ),
        box_threshold=0.10,
        text_threshold=0.20,
        tile_bias="small",
        max_side_ratio=0.08,
        max_area_ratio=0.006,
        min_component_area=18,
        nms_iou=0.28,
        fuse_iou=0.42,
    ),
    "mold": PromptSpec(
        key="mold",
        prompt=(
            "mold, wall mold, black mold, green mold, mildew, moss, algae, stain, "
            "water stain, dirty stain, discoloration, dark patch, green patch, damp area, moisture stain"
        ),
        box_threshold=0.10,
        text_threshold=0.20,
        tile_bias="medium",
        max_side_ratio=0.12,
        max_area_ratio=0.016,
        min_component_area=32,
        nms_iou=0.30,
        fuse_iou=0.45,
    ),
    "spall": PromptSpec(
        key="spall",
        prompt=(
            "spalling, concrete spalling, broken concrete, chipped concrete, peeling paint, "
            "flaking paint, exposed concrete, rough damaged surface, hole, cavity, surface loss, "
            "crumbling wall, missing material"
        ),
        box_threshold=0.20,
        text_threshold=0.20,
        tile_bias="medium",
        max_side_ratio=0.10,
        max_area_ratio=0.012,
        min_component_area=28,
        nms_iou=0.30,
        fuse_iou=0.45,
    ),
}


PROMPT_ORDER = ("crack", "mold", "spall")
