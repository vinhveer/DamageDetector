"""Guard that the JS EXPORT_MAP mirror stays in sync with the Python taxonomy.

The Electron app uses a static export_label map (src/features/labeling/exportLabel.js)
for instant UI updates. This test parses that JS file and asserts it equals the
default build_label_taxonomy().export_mapping, so any taxonomy change fails here.
"""
from __future__ import annotations

import re
from pathlib import Path

from shared.taxonomy.label_taxonomy import build_label_taxonomy

JS_PATH = Path(__file__).resolve().parents[1] / "app" / "src" / "features" / "labeling" / "exportLabel.js"


def _parse_js_export_map() -> dict[str, str]:
    text = JS_PATH.read_text(encoding="utf-8")
    block = re.search(r"EXPORT_MAP\s*=\s*\{(.*?)\}", text, re.DOTALL)
    assert block, "EXPORT_MAP literal not found in exportLabel.js"
    pairs = re.findall(r"(\w+)\s*:\s*'([^']+)'", block.group(1))
    return {k: v for k, v in pairs}


def test_js_export_map_matches_python_taxonomy():
    expected = build_label_taxonomy().export_mapping
    actual = _parse_js_export_map()
    assert actual == expected, f"JS EXPORT_MAP drifted from taxonomy: {actual} != {expected}"
