from __future__ import annotations


POS_PROMPTS: dict[str, list[str]] = {
    "crack": [
        "a close-up photo of an open crack line on a concrete surface",
        "one or more visible narrow open lines splitting the surface",
        "a long dark fissure line with clear edges on concrete or plaster",
        "multiple hairline cracks running across a building surface",
        "a jagged linear opening on a wall, slab, or concrete surface",
    ],
    "mold": [
        "a flat dirty stained patch on a wall or concrete surface",
        "a flat mold or mildew discoloration area without broken material",
        "green moss or algae on a flat building surface",
        "dark dirt or water stain on a flat concrete or plaster surface",
        "a flat discolored patch with soft blurry edges and no rough breakage",
    ],
    "spall": [
        "a close-up photo of chipped concrete with missing surface material",
        "a broken or flaked concrete area with rough uneven texture",
        "spalling damage with exposed inner material and jagged edges",
        "a rough damaged patch where the surface has peeled or broken off",
        "a bumpy uneven concrete surface with visible material loss",
    ],
    "other": [
        "an undamaged building surface with no crack mold or spalling damage",
        "a normal wall or concrete area that does not belong to crack mold or spall",
        "a shadow edge object joint tile line or background area that is not damage",
        "a blurry crop or irrelevant object not showing building surface damage",
        "a flat clean surface or ordinary texture without open lines stains or broken material",
    ],
}


def build_prompt_index(prompt_groups: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    labels: list[str] = []
    prompts: list[str] = []
    for label, variants in prompt_groups.items():
        for prompt in variants:
            text = str(prompt).strip()
            if not text:
                continue
            labels.append(str(label))
            prompts.append(text)
    return labels, prompts


# C5: negative anchors (penalize visually similar non-damage features) and per-label
# negative weights. Labels without a configured alpha default to 0.0 (no penalty).
NEGATIVE_ANCHORS: dict[str, list[str]] = {
    "crack": [
        "a smooth undamaged concrete surface with no cracks",
        "a dark shadow line that is not a crack",
        "an expansion joint, tile grout line, or panel seam",
        "a cable, wire, or pipe running along a wall",
    ],
    "mold": [
        "a clean dry concrete surface",
        "a rough broken concrete patch with missing material",
        "a thin open crack line",
    ],
    "spall": [
        "an intact smooth concrete surface with no missing material",
        "a flat surface stain without missing material",
        "a dark patch or shadow on concrete",
    ],
    "other": [
        "a clear crack line on concrete",
        "a flat mold or dirty stain patch",
        "a rough chipped spalling concrete patch",
    ],
}

NEGATIVE_ALPHA: dict[str, float] = {
    "crack": 1.0,
    "mold": 1.0,
    "spall": 1.0,
    "other": 0.5,
}
