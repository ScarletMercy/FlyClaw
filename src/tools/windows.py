"""Windows desktop automation tools via pyautogui + RapidOCR."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

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

_ocr_engine = None


def _get_ocr():
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine
    try:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
        logger.info("RapidOCR engine initialized")
        return _ocr_engine
    except Exception as e:
        logger.error("Failed to init OCR engine: %s", e)
        return None


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


async def windows_click(x: int, y: int, button: str = "left", clicks: int = 1) -> str:
    """Click at the specified screen coordinates.

    Args:
        x: X coordinate (pixels from left).
        y: Y coordinate (pixels from top).
        button: Mouse button: "left", "right", or "middle". Default "left".
        clicks: Number of clicks. Default 1 (set 2 for double-click).
    """
    if pyautogui is None:
        return "Error: pyautogui not available on this platform."
    try:
        await _run_sync(pyautogui.click, x, y, clicks=clicks, button=button)
        return f"Clicked at ({x}, {y}) button={button} clicks={clicks}"
    except Exception as e:
        return f"Error clicking: {e}"


async def windows_type(text: str) -> str:
    """Type text at the current cursor position.

    Args:
        text: The text string to type.
    """
    if pyautogui is None:
        return "Error: pyautogui not available on this platform."
    try:
        await _run_sync(pyautogui.typewrite, text, interval=0.02)
        return f"Typed: {text[:100]}"
    except Exception as e:
        return f"Error typing: {e}"


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
        # Support both "ctrl,c" and "ctrl+c" formats
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


async def windows_move(x: int, y: int) -> str:
    """Move the mouse cursor to the specified screen coordinates.

    Args:
        x: X coordinate (pixels from left).
        y: Y coordinate (pixels from top).
    """
    if pyautogui is None:
        return "Error: pyautogui not available on this platform."
    try:
        await _run_sync(pyautogui.moveTo, x, y, duration=0.3)
        return f"Moved to ({x}, {y})"
    except Exception as e:
        return f"Error moving mouse: {e}"


async def windows_drag(x: int, y: int, x2: int, y2: int) -> str:
    """Drag from one point to another (press, move, release).

    Args:
        x: Start X coordinate.
        y: Start Y coordinate.
        x2: End X coordinate.
        y2: End Y coordinate.
    """
    if pyautogui is None:
        return "Error: pyautogui not available on this platform."
    try:
        await _run_sync(pyautogui.moveTo, x, y, duration=0.1)
        await _run_sync(pyautogui.drag, x2 - x, y2 - y, duration=0.5)
        return f"Dragged from ({x}, {y}) to ({x2}, {y2})"
    except Exception as e:
        return f"Error dragging: {e}"


async def windows_ocr(region: str = "") -> str:
    """OCR the screen or a region, returning text blocks with bounding box coordinates.

    Useful for locating UI elements by text. Returns each block with position (x, y, w, h) and confidence.
    Combine with windows_click to click on found text.

    Args:
        region: Optional region to OCR in "x,y,w,h" format. Default: full screen.
    """
    if pyautogui is None:
        return "Error: pyautogui not available on this platform."

    try:
        if region:
            parts = [int(p.strip()) for p in region.split(",")]
            if len(parts) != 4:
                return "Error: region must be 'x,y,w,h' format."
            rx, ry, rw, rh = parts
            img = await _run_sync(pyautogui.screenshot, region=(rx, ry, rw, rh))
        else:
            img = await _run_sync(pyautogui.screenshot)

        ocr = _get_ocr()
        if ocr is None:
            return "Error: OCR engine not available. Install with: uv pip install rapidocr-onnxruntime"

        def _run_ocr():
            return ocr(img)

        result, elapse = await _run_sync(_run_ocr)

        lines = []
        idx = 0
        if result:
            for item in result:
                box, text, conf = item
                if conf < 0.5:
                    continue
                idx += 1
                x_min = int(min(p[0] for p in box))
                y_min = int(min(p[1] for p in box))
                x_max = int(max(p[0] for p in box))
                y_max = int(max(p[1] for p in box))
                w = x_max - x_min
                h = y_max - y_min
                if region:
                    x_min += rx
                    y_min += ry
                lines.append(f"[{idx}] \"{text}\" at ({x_min},{y_min},{w},{h}) conf={conf:.2f}")

        if not lines:
            return "No text found on screen."
        # Save full results to file for reference
        import json as _json
        from src.tools.file_tools import _BASE_DIR
        _out = [{"text": item[1], "conf": float(item[2]),
                 "box": [[int(p[0]), int(p[1])] for p in item[0]]}
                for item in result if float(item[2]) >= 0.5]
        _path = str(Path(_BASE_DIR) / "ocr_result.json")
        with open(_path, "w", encoding="utf-8") as _f:
            _json.dump(_out, _f, ensure_ascii=False)
        header = f"Found {len(lines)} text blocks (full results saved to ocr_result.json):"
        return header + "\n" + "\n".join(lines)
    except Exception as e:
        return f"Error during OCR: {e}"


def get_tools() -> list:
    from src.agent.tooldef import ToolDef

    if pyautogui is None:
        return []

    return [
        ToolDef.from_function(windows_screenshot),
        ToolDef.from_function(windows_screen_size),
        ToolDef.from_function(windows_click),
        ToolDef.from_function(windows_type),
        ToolDef.from_function(windows_press),
        ToolDef.from_function(windows_hotkey),
        ToolDef.from_function(windows_scroll),
        ToolDef.from_function(windows_move),
        ToolDef.from_function(windows_drag),
        ToolDef.from_function(windows_ocr),
    ]
