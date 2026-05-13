"""Load workspace bootstrap context files (AGENTS.md, SOUL.md, IDENTITY.md, etc.)."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("myclaw.bootstrap")

DEFAULT_BOOTSTRAP_FILES = [
    "AGENTS.md",
    "SOUL.md",
    "IDENTITY.md",
    "USER.md",
    "HEARTBEAT.md",
]


def load_bootstrap_files(
    workspace_dir: str,
    extra_names: list[str] | None = None,
) -> list[dict]:
    """Scan workspace directory and load existing bootstrap files.

    Returns a list of {"path": name, "content": text} dicts for files that exist.
    """
    names = list(DEFAULT_BOOTSTRAP_FILES)
    if extra_names:
        for n in extra_names:
            if n not in names:
                names.append(n)

    files: list[dict] = []
    workspace = Path(workspace_dir).expanduser().resolve()

    for name in names:
        path = workspace / name
        if path.is_file():
            try:
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    files.append({"path": name, "content": content})
                    logger.info("Loaded bootstrap file: %s (%d chars)", name, len(content))
            except Exception as e:
                logger.warning("Failed to read bootstrap file %s: %s", name, e)

    return files
