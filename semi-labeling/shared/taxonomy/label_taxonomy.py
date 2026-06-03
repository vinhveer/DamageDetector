from __future__ import annotations

import json
from dataclasses import dataclass


WORKING_LABELS: tuple[str, ...] = ("crack", "mold", "spall", "reject")
DAMAGE_LABELS: tuple[str, ...] = ("crack", "mold", "spall")
REJECT_LABELS: tuple[str, ...] = ("other", "shadow", "edge", "background", "object", "unknown", "reject")

GUIDELINES: dict[str, str] = {
    "crack": "dai, manh, tuyen tinh, bien ro",
    "mold": "loang mau, mem bien, thuong xanh/den/nau",
    "spall": "bong/vo/mat vat lieu, texture tho",
    "reject": "khong thuoc crack/mold/spall hoac qua mo ho",
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
        return "reject"

    def export_label(self, label: str) -> str:
        normalized = self.normalize_label(label)
        return self.export_mapping.get(normalized, "reject")


def build_label_taxonomy(*, version_id: str = "label_taxonomy_v1", stain_export_label: str = "reject") -> LabelTaxonomy:
    _ = stain_export_label  # Backward-compatible, ignored by the 3-class + reject taxonomy.
    export_mapping = {
        "crack": "crack",
        "mold": "mold",
        "spall": "spall",
        "other": "reject",
        "stain": "reject",
        "efflorescence": "reject",
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
