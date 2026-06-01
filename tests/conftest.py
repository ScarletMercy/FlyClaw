import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── aiosqlite creates non-daemon threads that block process exit ──
# Patch it at import time so every connection uses a daemon thread.
import aiosqlite.core as _aiosqlite_core


class _DaemonThread(threading.Thread):
    """Thread subclass that always sets daemon=True."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.daemon = True


_aiosqlite_core.Thread = _DaemonThread
