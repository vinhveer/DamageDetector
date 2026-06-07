from __future__ import annotations

# Prompt mặc định cho bước cắt nhà.
#   positive → điểm DƯƠNG cho SAM (thuộc nhà)
#   negative → điểm ÂM cho SAM   (KHÔNG thuộc nhà: cửa sổ / cửa đi)
DEFAULT_POSITIVE_QUERIES: tuple[str, ...] = (
    "whole house",
    "entire house",
    "house facade",
    "building facade",
    "residential building facade",
    "front of house",
    "exterior wall",
    "concrete wall facade",
)

DEFAULT_NEGATIVE_QUERIES: tuple[str, ...] = (
    "window",
    "windows",
    "glass window",
    "window frame",
    "window opening",
    "door",
    "doors",
    "glass door",
    "door frame",
    "door opening",
    "entrance door",
    "balcony door",
    "glass pane",
)

# Từ khoá phân loại một GDINO label về vai trò negative. Bất kỳ label nào khớp
# (substring) sẽ được xem là negative; còn lại là positive (house).
_NEGATIVE_KEYWORDS: tuple[str, ...] = (
    "window",
    "door",
    "glass",
    "entrance",
    "opening",
    "pane",
)


def normalize_queries(values, fallback: tuple[str, ...]) -> list[str]:
    """Chuẩn hoá danh sách query từ CLI; rỗng → fallback mặc định."""
    out: list[str] = []
    seen: set[str] = set()
    for v in (values or []):
        text = str(v or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
    if not out:
        return list(fallback)
    return out


def combined_text_queries(positive: list[str], negative: list[str]) -> list[str]:
    """Gộp positive ∪ negative để gửi một lần cho GDINO (loại trùng, giữ thứ tự)."""
    merged: list[str] = []
    seen: set[str] = set()
    for q in list(positive) + list(negative):
        key = str(q).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(q)
    return merged


def role_for_label(label: str) -> str:
    """Map một GDINO label thô về 'negative' (window/door) hoặc 'house'."""
    text = str(label or "").strip().lower()
    if not text:
        return "house"
    for kw in _NEGATIVE_KEYWORDS:
        if kw in text:
            return "negative"
    return "house"
