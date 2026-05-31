from __future__ import annotations

from dataclasses import dataclass


DEFAULT_PROMPT_GROUPS: list[tuple[str, str]] = [
    (
        "crack",
        "crack, surface crack, wall crack, concrete crack, thin crack, long crack, hairline crack, fracture line",
    ),
    (
        "mold",
        "mold, mildew, moss, algae stain, water stain, damp stain, dirty stain, discolored patch, dark stain on wall, surface contamination",
    ),
    (
        "spall",
        "spalling, spalled concrete, surface spall, broken concrete, chipped concrete, flaking surface, peeling surface, surface loss, missing material, damaged concrete surface",
    ),
]


@dataclass(frozen=True)
class PromptGroup:
    group_id: int
    name: str
    slug: str
    prompt_text: str
    queries: tuple[str, ...]


def _slugify(text: str) -> str:
    parts: list[str] = []
    prev_sep = False
    for ch in str(text or "").strip().lower():
        if ch.isalnum():
            parts.append(ch)
            prev_sep = False
        elif not prev_sep:
            parts.append("-")
            prev_sep = True
    slug = "".join(parts).strip("-")
    return slug or "group"


def parse_prompt_group(value: str, index: int) -> PromptGroup:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Prompt group must not be empty.")
    if "=" in raw:
        name, prompt_text = raw.split("=", 1)
    elif "::" in raw:
        name, prompt_text = raw.split("::", 1)
    else:
        name = f"group_{index:02d}"
        prompt_text = raw
    name = str(name).strip() or f"group_{index:02d}"
    queries = tuple(q.strip() for q in str(prompt_text).split(",") if q.strip())
    if not queries:
        raise ValueError(f"Prompt group '{name}' has no valid queries.")
    return PromptGroup(
        group_id=index,
        name=name,
        slug=f"{index:02d}-{_slugify(name)}",
        prompt_text=", ".join(queries),
        queries=queries,
    )


def load_prompt_groups(raw_values: list[str] | tuple[str, ...] | None = None) -> list[PromptGroup]:
    raw_values = list(raw_values or [])
    if not raw_values:
        raw_values = [f"{name}={prompt}" for name, prompt in DEFAULT_PROMPT_GROUPS]
    return [parse_prompt_group(value, idx) for idx, value in enumerate(raw_values, start=1)]


def parse_request_names(raw: str) -> list[str]:
    names = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    seen: set[str] = set()
    out: list[str] = []
    for item in names:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out or ["dino", "groundingdino"]


def combined_queries(prompt_groups: list[PromptGroup]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in prompt_groups:
        for query in group.queries:
            key = query.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(query)
    return merged


def match_prompt_groups(label: str, prompt_groups: list[PromptGroup]) -> list[int]:
    text = str(label or "").strip().lower()
    if not text:
        return []
    matched: list[int] = []
    for group in prompt_groups:
        for query in group.queries:
            q = query.strip().lower()
            if q and q in text:
                matched.append(int(group.group_id))
                break
    return matched
