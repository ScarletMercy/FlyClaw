from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from src.agent.tooldef import ToolDef

logger = logging.getLogger("flyclaw.canvas.tool")


async def canvas_render(text: str, format: str = "text", surface_id: str = "main") -> str:
    """Render content on the canvas. Use this to display rich output to the user.

    Args:
        text: Content to render.
        format: Rendering format — "text", "markdown", "json", "html".
        surface_id: Target surface ID (default: "main").
    """
    from src.canvas.a2ui import A2uiBuilder

    builder = A2uiBuilder()
    if format == "markdown":
        builder.add_markdown(text, surface_id=surface_id)
    elif format in ("json", "html"):
        builder.add_text(text, usage_hint="body", surface_id=surface_id)
    else:
        builder.add_text(text, surface_id=surface_id)

    jsonl = builder.to_jsonl()

    try:
        from src.canvas.server import _root, broadcast_reload

        if _root:
            out_path = _root / f"__render_{surface_id}.jsonl"
            await asyncio.to_thread(out_path.write_text, "\n".join(jsonl), "utf-8")

        asyncio.create_task(_safe_broadcast())
    except Exception as exc:
        logger.debug("canvas_render broadcast failed: %s", exc)

    return f"Canvas rendered on surface '{surface_id}'"


async def _safe_broadcast():
    try:
        from src.canvas.server import broadcast_reload
        await broadcast_reload()
    except Exception:
        pass


async def canvas_navigate(url: str) -> str:
    """Navigate the canvas to a URL.

    Args:
        url: URL to navigate to.
    """
    try:
        from src.canvas.server import _ws_clients

        msg = json.dumps({"navigate": {"url": url}})
        dead = set()
        for ws in list(_ws_clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        _ws_clients.difference_update(dead)
    except Exception as exc:
        logger.debug("canvas_navigate failed: %s", exc)

    return f"Canvas navigated to: {url}"


async def canvas_eval(java_script: str) -> str:
    """Execute JavaScript in the canvas WebView.

    Args:
        java_script: JavaScript code to execute.
    """
    try:
        from src.canvas.server import _ws_clients

        msg = json.dumps({"eval": {"script": java_script}})
        dead = set()
        for ws in list(_ws_clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        _ws_clients.difference_update(dead)
    except Exception as exc:
        logger.debug("canvas_eval failed: %s", exc)

    return f"Canvas eval executed ({len(java_script)} chars)"


def get_canvas_tools() -> list[ToolDef]:
    return [
        ToolDef.from_function(canvas_render),
        ToolDef.from_function(canvas_navigate),
        ToolDef.from_function(canvas_eval),
    ]
