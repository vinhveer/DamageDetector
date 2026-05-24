from __future__ import annotations


DEFAULT_LABEL_PROMPTS: dict[str, list[str]] = {
    "crack": [
        "a close-up photo of a thin narrow crack line on concrete surface",
        "a long dark fracture line with sharp edges on a wall",
        "a fine irregular hairline crack on plaster or concrete",
        "a linear split with clear boundaries in building material",
        "a jagged crack line running across a surface",
    ],
    "mold": [
        "a close-up photo of mold growing on a concrete surface",
        "a dark green mold or mildew patch on a wall",
        "moss or algae covering a damp surface",
        "biological growth on outdoor concrete",
        "a fuzzy organic colony of mildew on a wall",
        "lichen patches on a stone surface",
    ],
    "stain": [
        "a close-up photo of a dark water stain on concrete",
        "a rust-colored stain leaking down a wall",
        "a damp patch discoloration without sharp boundaries",
        "a dirty smudge on a concrete surface",
        "a soiling mark caused by water flow on a wall",
        "a brown or grey discoloration patch on concrete",
    ],
}


def build_prompt_index(prompt_groups: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    labels: list[str] = []
    texts: list[str] = []
    for label, variants in prompt_groups.items():
        for v in variants:
            t = str(v).strip()
            if not t:
                continue
            labels.append(str(label))
            texts.append(t)
    return labels, texts
