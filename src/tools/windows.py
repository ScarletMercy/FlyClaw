"""Windows desktop automation tools via pyautogui."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger("myclaw.windows")

if sys.platform == "win32":
    try:
        import pyautogui

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.1
    except ImportError:
        pyautogui = None  # type: ignore[assignment]
        logger.warning("pyautogui not installed, windows tools will be unavailable")
else:
    pyautogui = None  # type: ignore[assignment]


def _screenshot_dir() -> str:
    d = Path.home() / ".myclaw" / "data" / "screenshots"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


async def _run_sync(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


async def windows_screenshot() -> str:
    """Take a screenshot of the entire screen. Returns the saved file path.

    The screenshot can be used with media_understanding tools to let AI see the screen.
    """
    if pyautogui is None:
        return "Error: pyautogui not available on this platform."
    try:
        path = os.path.join(_screenshot_dir(), f"screen_{int(time.time())}.png")
        await _run_sync(pyautogui.screenshot, path)
        return f"Screenshot saved: {path}"
    except Exception as e:
        return f"Error taking screenshot: {e}"


async def windows_screen_size() -> str:
    """Get the screen resolution. Returns width x height in pixels."""
    if pyautogui is None:
        return "Error: pyautogui not available on this platform."
    try:
        size = await _run_sync(lambda: pyautogui.size())
        return f"{size.width}x{size.height}"
    except Exception as e:
        return f"Error getting screen size: {e}"


async def windows_press(key: str) -> str:
    """Press a keyboard key (e.g. Enter, Tab, Escape, Backspace, delete, up, down, left, right).

    Args:
        key: Key name (e.g. "enter", "tab", "escape", "backspace", "delete", "up", "down", "left", "right", "space", "win", "capslock").
    """
    if pyautogui is None:
        return "Error: pyautogui not available on this platform."
    try:
        await _run_sync(pyautogui.press, key)
        return f"Pressed: {key}"
    except Exception as e:
        return f"Error pressing key: {e}"


async def windows_hotkey(keys: str) -> str:
    """Press a keyboard shortcut / combination of keys held simultaneously.

    Args:
        keys: Comma-separated key names to press together. E.g. "ctrl,c" for Ctrl+C, "alt,tab" for Alt+Tab, "ctrl,shift,esc" for Ctrl+Shift+Esc.
    """
    if pyautogui is None:
        return "Error: pyautogui not available on this platform."
    try:
        if "+" in keys:
            key_list = [k.strip() for k in keys.split("+")]
        else:
            key_list = [k.strip() for k in keys.split(",")]
        await _run_sync(pyautogui.hotkey, *key_list)
        return f"Hotkey: {'+'.join(key_list)}"
    except Exception as e:
        return f"Error pressing hotkey: {e}"


async def windows_scroll(x: int = 0, y: int = 0, amount: int = 3) -> str:
    """Scroll the mouse wheel at a position.

    Args:
        x: X coordinate. Default 0 (current position).
        y: Y coordinate. Default 0 (current position).
        amount: Number of scroll clicks. Positive = scroll up, negative = scroll down. Default 3.
    """
    if pyautogui is None:
        return "Error: pyautogui not available on this platform."
    try:
        if x != 0 or y != 0:
            await _run_sync(pyautogui.scroll, amount, x, y)
        else:
            await _run_sync(pyautogui.scroll, amount)
        return f"Scrolled amount={amount} at ({x}, {y})"
    except Exception as e:
        return f"Error scrolling: {e}"


def get_tools() -> list:
    from src.agent.tooldef import ToolDef

    if pyautogui is None:
        return []

    return [
        ToolDef.from_function(windows_screenshot),
        ToolDef.from_function(windows_screen_size),
        ToolDef.from_function(windows_press),
        ToolDef.from_function(windows_hotkey),
        ToolDef.from_function(windows_scroll),
    ]
