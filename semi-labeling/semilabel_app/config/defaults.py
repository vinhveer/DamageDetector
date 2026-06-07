from __future__ import annotations

from typing import Any

from ..paths import default_export_dir, default_image_root, default_resemi_db


LABELS = ["crack", "mold", "spall", "reject"]
PROTOTYPE_LABELS = ["crack", "mold", "spall", "reject"]
DEFAULT_RUN_ID = "myrun"
DEFAULT_MODEL_NAME = "facebook/dinov2-giant"
DEFAULT_VIEW_NAME = "tight"
DEFAULT_SHORTCUTS: dict[str, str] = {
    "review_label_1": "1",
    "review_label_2": "2",
    "review_label_3": "3",
    "review_label_4": "4",
    "next_item": "Space",
    "next_item_alt": "Down",
    "previous_item": "Up",
    "undo": "Z",
    "save": "Ctrl+S",
    "prototype_pick": "Enter",
    "prototype_reject": "R",
    "prototype_unpick": "U",
    "tab_review": "Ctrl+1",
    "tab_qa": "Ctrl+2",
    "tab_images": "Ctrl+3",
    "tab_prototype": "Ctrl+4",
}

DEFAULT_SETTINGS: dict[str, Any] = {
    "db_path": str(default_resemi_db()),
    "image_root": str(default_image_root()),
    "run_id": DEFAULT_RUN_ID,
    "labels": LABELS,
    "prototype_labels": PROTOTYPE_LABELS,
    "model_name": DEFAULT_MODEL_NAME,
    "view_name": DEFAULT_VIEW_NAME,
    "reject_below": 0.5,
    "per_band": 200,
    "sample_percent": 10,
    "cleaned_limit": 500,
    "export_dir": str(default_export_dir()),
    "export_format": "yolo",
    "shortcuts": DEFAULT_SHORTCUTS,
}


def migrate_settings(raw: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(raw)
    if "resemiDbPath" in migrated and "db_path" not in migrated:
        migrated["db_path"] = migrated.pop("resemiDbPath")
    if "imageRootPath" in migrated and "image_root" not in migrated:
        migrated["image_root"] = migrated.pop("imageRootPath")
    if str(migrated.get("model_name") or "").strip() == "facebook/dinov2-small":
        migrated["model_name"] = DEFAULT_MODEL_NAME
    migrated["labels"] = [label for label in migrated.get("labels", LABELS) if label in LABELS]
    if not migrated["labels"]:
        migrated["labels"] = LABELS
    migrated["prototype_labels"] = [
        label for label in migrated.get("prototype_labels", PROTOTYPE_LABELS) if label in PROTOTYPE_LABELS
    ]
    if not migrated["prototype_labels"]:
        migrated["prototype_labels"] = PROTOTYPE_LABELS
    return migrated
