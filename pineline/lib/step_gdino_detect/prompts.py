from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptGroup:
    group_id: int
    name: str
    slug: str
    queries: tuple[str, ...]


DEFAULT_DAMAGE_PROMPT_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    (
        "crack",
        (
            "crack", "surface crack", "concrete crack", "wall crack",
            "thin crack", "long crack", "hairline crack", "fracture line",
        ),
    ),
    (
        "mold",
        (
            "mold", "mildew", "moss", "algae", "algae stain",
            "biological growth on wall", "green stain on concrete",
            "fungal growth", "lichen on surface",
        ),
    ),
    (
        "stain",
        (
            "water stain", "damp stain", "dark stain on wall",
            "discoloration patch", "dirty stain", "rust stain",
            "surface contamination", "blackish patch on concrete",
        ),
    ),
    (
        "spall",
        (
            "spall", "spalling", "concrete spalling", "surface spall",
            "flaked concrete", "chipped concrete", "delamination",
        ),
    ),
]


def default_prompt_groups() -> list[PromptGroup]:
    out: list[PromptGroup] = []
    for idx, (name, queries) in enumerate(DEFAULT_DAMAGE_PROMPT_GROUPS, start=1):
        out.append(
            PromptGroup(
                group_id=idx,
                name=name,
                slug=f"{idx:02d}-{name}",
                queries=tuple(queries),
            )
        )
    return out


def parse_prompt_groups(raw_values: list[str] | None) -> list[PromptGroup]:
    """Parse 'name=q1,q2,...' style values; fall back to defaults if empty."""
    raw_values = list(raw_values or [])
    if not raw_values:
        return default_prompt_groups()
    out: list[PromptGroup] = []
    for idx, value in enumerate(raw_values, start=1):
        text = str(value or "").strip()
        if not text:
            continue
        if "=" in text:
            name, body = text.split("=", 1)
        else:
            name = f"group_{idx:02d}"
            body = text
        queries = tuple(q.strip() for q in body.split(",") if q.strip())
        if not queries:
            continue
        out.append(
            PromptGroup(
                group_id=idx,
                name=name.strip() or f"group_{idx:02d}",
                slug=f"{idx:02d}-{name.strip().lower()}",
                queries=queries,
            )
        )
    return out or default_prompt_groups()


def combined_queries(groups: list[PromptGroup]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for g in groups:
        for q in g.queries:
            key = q.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(q)
    return merged


def match_group_for_label(label: str, groups: list[PromptGroup]) -> tuple[int, str] | None:
    """Map a raw GroundingDINO phrase back to one of the prompt groups.

    GDINO returns free-form phrases (e.g. "hairline crack", "green stain on
    concrete"). We collapse them to the single canonical group name
    (crack/mold/stain). Matching is done longest-query-first so that a more
    specific phrase wins. A keyword fallback handles morphological variants
    (e.g. "cracks") but only on *distinctive* damage words, never on generic
    words like "wall", "surface" or "concrete".
    """
    text = str(label or "").strip().lower()
    if not text:
        return None

    # Distinctive keywords per group. Generic scene words (wall, surface,
    # concrete, patch, growth, line, on, ...) are deliberately excluded so a
    # bare "wall" does not get tagged as crack.
    keywords = {
        "crack": ("crack", "cracks", "cracking", "fracture", "fissure", "hairline"),
        "mold": ("mold", "mould", "mildew", "moss", "algae", "fungal", "fungus",
                 "lichen", "biological"),
        "stain": ("stain", "stains", "discoloration", "discolouration",
                  "damp", "rust", "contamination"),
        "spall": ("spall", "spalling", "flaked", "chipped", "delamination"),
    }
    name_to_group = {g.name.strip().lower(): g.group_id for g in groups}
    tokens = set(text.replace("-", " ").split())

    # 1) direct substring match against full query phrases, longest wins.
    best: tuple[int, str, int] | None = None
    for g in groups:
        for q in g.queries:
            qq = q.strip().lower()
            if qq and qq in text:
                if best is None or len(qq) > best[2]:
                    best = (g.group_id, g.name, len(qq))
    if best is not None:
        return best[0], best[1]

    # 2) distinctive keyword token match (handles plurals / merged phrases).
    for g in groups:
        kw = keywords.get(g.name.strip().lower(), (g.name.strip().lower(),))
        for k in kw:
            if k in tokens or any(k in tok for tok in tokens):
                return g.group_id, g.name

    # 3) the exact group name appears anywhere in the phrase.
    for name, gid in name_to_group.items():
        if name and name in text:
            return gid, name
    return None
