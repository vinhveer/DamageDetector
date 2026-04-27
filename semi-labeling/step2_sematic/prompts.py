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
