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
    text = str(label or "").strip().lower()
    if not text:
        return None
    for g in groups:
        for q in g.queries:
            qq = q.strip().lower()
            if qq and qq in text:
                return g.group_id, g.name
    return None
