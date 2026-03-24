from __future__ import annotations

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
