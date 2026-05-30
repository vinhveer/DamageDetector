from __future__ import annotations


POS_PROMPTS: dict[str, list[str]] = {
    "crack": [
        "a close-up photo of a thin narrow crack line on concrete surface",
        "a long dark fracture line with sharp edges on a wall",
        "a fine irregular hairline crack on plaster or concrete",
        "a linear split with clear boundaries in building material",
        "a jagged crack line running across a surface",
    ],
    "mold": [
        "a close-up photo of a mold stain patch on a wall surface",
        "a dark or green mold area with blurry edges",
        "a dirty discoloration patch without sharp lines",
        "mildew or moss growing on concrete surface",
        "an irregular stain area with soft boundaries",
    ],
    "spall": [
        "a close-up photo of broken concrete surface with missing material",
        "a chipped concrete area with rough texture",
        "spalling damage exposing inner material",
        "a hole or flaked region on concrete surface",
        "a damaged surface with pieces falling off",
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
        "a paint stain or watermark that is not mold",
        "a smooth shadow on the wall",
    ],
    "spall": [
        "an intact smooth concrete surface with no missing material",
        "a shallow surface stain without missing material",
        "a dark patch or shadow on concrete",
    ],
}

NEGATIVE_ALPHA: dict[str, float] = {
    "crack": 1.0,
    "mold": 1.0,
    "spall": 1.0,
}
