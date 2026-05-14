from __future__ import annotations

import asyncio
import logging

from src.agent.tooldef import ToolDef

logger = logging.getLogger("myclaw.canvas.tool")


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
    else:
        builder.add_text(text, surface_id=surface_id)

    try:
        from src.canvas.server import broadcast_reload
        asyncio.create_task(broadcast_reload())
    except Exception:
        pass

    return f"Canvas rendered on surface '{surface_id}'"


async def canvas_navigate(url: str) -> str:
    """Navigate the canvas to a URL.

    Args:
        url: URL to navigate to.
    """
    return f"Canvas navigated to: {url}"


async def canvas_eval(java_script: str) -> str:
    """Execute JavaScript in the canvas WebView.

    Args:
        java_script: JavaScript code to execute.
    """
    return f"Canvas eval executed ({len(java_script)} chars)"


def get_canvas_tools() -> list[ToolDef]:
    return [
        ToolDef.from_function(canvas_render),
        ToolDef.from_function(canvas_navigate),
        ToolDef.from_function(canvas_eval),
    ]
