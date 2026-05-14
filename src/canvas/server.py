from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

from src.canvas.file_resolver import FileResolver

logger = logging.getLogger("myclaw.canvas")

CANVAS_PATH = "/__myclaw__/canvas"
CANVAS_WS_PATH = "/__myclaw__/ws"

LIVE_RELOAD_SCRIPT = """<script>
(function() {
    const ws = new WebSocket((location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + location.host + '/__myclaw__/ws');
    ws.onmessage = function(e) { if (e.data === 'reload') location.reload(); };
    ws.onclose = function() { setTimeout(function() { location.reload(); }, 3000); };
})();
</script>"""

router = APIRouter()

_root: Path | None = None
_resolver: FileResolver | None = None
_ws_clients: set[WebSocket] = set()


def init_canvas(root: Path):
    global _root, _resolver
    _root = root.resolve()
    _root.mkdir(parents=True, exist_ok=True)
    _resolver = FileResolver(_root)
    logger.info("Canvas host initialized at %s", _root)


@router.get(f"{CANVAS_PATH}/{{path:path}}")
@router.get(CANVAS_PATH)
async def serve_canvas(path: str = "", request: Request = None):
    if not _resolver:
        return HTMLResponse("<h1>Canvas not initialized</h1>", 503)
    try:
        resolved, mime = _resolver.resolve(path)
        if resolved is None:
            return HTMLResponse(_default_page(), 200)
        return FileResponse(resolved, media_type=mime)
    except ValueError as e:
        logger.warning("Canvas path rejected: %s", e)
        return HTMLResponse(f"Forbidden: {e}", 403)


@router.websocket(CANVAS_WS_PATH)
async def canvas_ws(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)


async def broadcast_reload():
    dead = set()
    for ws in _ws_clients:
        try:
            await ws.send_text("reload")
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


def _default_page() -> str:
    return f"""<!DOCTYPE html>
<html><head><title>MyClaw Canvas</title>{LIVE_RELOAD_SCRIPT}</head>
<body><h1>MyClaw Canvas</h1><p>Drop files in the canvas root directory to get started.</p></body></html>"""
