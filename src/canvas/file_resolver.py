from __future__ import annotations

import mimetypes
from pathlib import Path


class FileResolver:
    def __init__(self, root: Path):
        self._root = root.resolve()

    def resolve(self, relative_path: str) -> tuple[Path | None, str]:
        clean = relative_path.strip("/")
        if ".." in clean.split("/"):
            raise ValueError(f"Path traversal rejected: {relative_path}")

        target = (self._root / clean).resolve() if clean else self._root / "index.html"

        try:
            target.relative_to(self._root)
        except ValueError:
            raise ValueError(f"Path escape rejected: {relative_path}")

        if target.is_dir():
            target = target / "index.html"

        if not target.exists():
            return None, ""

        if target.is_symlink():
            raise ValueError(f"Symlink rejected: {relative_path}")

        mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        return target, mime
