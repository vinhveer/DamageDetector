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
            "crack, wall crack, concrete crack, plaster crack, surface crack, "
            "thin crack, long crack, hairline crack, irregular crack, jagged crack, "
            "linear crack, fracture line, fissure, wall fissure, concrete fissure, "
            "dark crack line, vertical crack, horizontal crack, diagonal crack, branching crack"
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
            "mold, wall mold, black mold, green mold, mildew, mildew stain, "
            "moss, moss patch, algae, algae patch, stain, flat stain, water stain, "
            "dirty stain, dirty patch, discoloration, dark patch, green patch, brown stain, "
            "damp area, damp stain, humidity stain, moisture stain"
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
            "spalling, concrete spalling, concrete damage, broken concrete, chipped concrete, "
            "chipped wall, broken plaster, delamination, peeling paint, plaster peeling, "
            "flaking paint, flaked surface, exposed concrete, exposed aggregate, "
            "rough damaged surface, hole, cavity, surface loss, material loss, "
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
