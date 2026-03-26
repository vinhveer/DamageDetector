from __future__ import annotations

import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class FileService:
    def list_images(self, folder: str | Path) -> list[str]:
        root = Path(folder)
        if not root.is_dir():
            return []
        items = [
            str(path)
            for path in sorted(root.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        return items

    def import_images(self, source_folder: str | Path, target_folder: str | Path) -> list[str]:
        source_root = Path(source_folder)
        target_root = Path(target_folder)
        if not source_root.is_dir() or not target_root.is_dir():
            return []
        copied: list[str] = []
        for source_path in sorted(source_root.iterdir()):
            if not source_path.is_file() or source_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            target_path = self._unique_target_path(target_root, source_path.name)
            shutil.copy2(source_path, target_path)
            copied.append(str(target_path))
        return copied

    def _unique_target_path(self, folder: Path, name: str) -> Path:
        candidate = folder / name
        if not candidate.exists():
            return candidate
        stem = Path(name).stem
        suffix = Path(name).suffix
        index = 1
        while True:
            candidate = folder / f"{stem}_{index}{suffix}"
            if not candidate.exists():
                return candidate
            index += 1
