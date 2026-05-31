from __future__ import annotations

import json
from dataclasses import dataclass


WORKING_LABELS: tuple[str, ...] = (
    "crack",
    "mold",
    "spall",
    "stain",
    "efflorescence",
    "shadow",
    "edge",
    "background",
    "object",
    "unknown",
    "reject",
)

DAMAGE_LABELS: tuple[str, ...] = ("crack", "mold", "spall", "stain", "efflorescence")
REJECT_LABELS: tuple[str, ...] = ("shadow", "edge", "background", "object", "unknown", "reject")

GUIDELINES: dict[str, str] = {
    "crack": "dai, manh, tuyen tinh, bien ro",
    "mold": "loang mau, mem bien, thuong xanh/den/nau",
    "spall": "bong/vo/mat vat lieu, texture tho",
    "stain": "doi mau nhung khong ro nam moc",
    "reject": "khong phai hu hong hoac qua mo ho",
}


@dataclass(frozen=True)
class LabelTaxonomy:
    version_id: str
    working_labels: tuple[str, ...]
    damage_labels: tuple[str, ...]
    reject_labels: tuple[str, ...]
    export_mapping: dict[str, str]
    guidelines: dict[str, str]

    @property
    def working_labels_json(self) -> str:
        return json.dumps(list(self.working_labels), ensure_ascii=False, sort_keys=True)

    @property
    def damage_labels_json(self) -> str:
        return json.dumps(list(self.damage_labels), ensure_ascii=False, sort_keys=True)

    @property
    def reject_labels_json(self) -> str:
        return json.dumps(list(self.reject_labels), ensure_ascii=False, sort_keys=True)

    @property
    def export_mapping_json(self) -> str:
        return json.dumps(self.export_mapping, ensure_ascii=False, sort_keys=True)

    @property
    def guidelines_json(self) -> str:
        return json.dumps(self.guidelines, ensure_ascii=False, sort_keys=True)

    def normalize_label(self, label: str) -> str:
        raw = str(label or "").strip().lower()
        if raw in self.working_labels:
            return raw
        return "unknown"

    def export_label(self, label: str) -> str:
        normalized = self.normalize_label(label)
        return self.export_mapping.get(normalized, "reject")


def build_label_taxonomy(*, version_id: str = "label_taxonomy_v1", stain_export_label: str = "stain") -> LabelTaxonomy:
    stain_target = str(stain_export_label or "stain").strip().lower()
    if stain_target not in {"stain", "mold", "reject"}:
        raise ValueError("stain_export_label must be one of: stain, mold, reject")
    export_mapping = {
        "crack": "crack",
        "mold": "mold",
        "spall": "spall",
        "stain": stain_target,
        "efflorescence": "stain",
        "shadow": "reject",
        "edge": "reject",
        "background": "reject",
        "object": "reject",
        "unknown": "reject",
        "reject": "reject",
    }
    return LabelTaxonomy(
        version_id=version_id,
        working_labels=WORKING_LABELS,
        damage_labels=DAMAGE_LABELS,
        reject_labels=REJECT_LABELS,
        export_mapping=export_mapping,
        guidelines=GUIDELINES,
    )
