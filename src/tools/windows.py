"""Windows desktop automation tools via pyautogui."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger("flyclaw.windows")

if sys.platform == "win32":
    try:
        import pyautogui

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.1
    except ImportError:
        pyautogui = None  # type: ignore[assignment]
        logger.warning("pyautogui not installed, windows tools will be unavailable")
    try:
        from PIL import ImageGrab
    except ImportError:
        ImageGrab = None  # type: ignore[assignment]
else:
    pyautogui = None  # type: ignore[assignment]
    ImageGrab = None  # type: ignore[assignment]


def _screenshot_dir() -> str:
    from src.config import load_config

    cfg = load_config()
    workspace = Path(cfg.agents.workspace).expanduser().resolve()
    d = workspace / "screenshots"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


async def windows_screenshot() -> str:
    """Take a screenshot of the entire screen. Returns the saved file path.

    The screenshot can be used with media_understanding tools to let AI see the screen.
    """
    if ImageGrab is None:
        return "Error: Pillow not available on this platform."
    try:
        path = os.path.join(_screenshot_dir(), f"screen_{int(time.time())}.png")

        def _grab():
            img = ImageGrab.grab()
            img.save(path)

        await asyncio.to_thread(_grab)
        return f"Screenshot saved: {path}"
    except Exception as e:
        return f"Error taking screenshot: {e}"


async def windows_press(key: str) -> str:
    """Press a keyboard key (e.g. Enter, Tab, Escape, Backspace, delete, up, down, left, right).

    Args:
        key: Key name (e.g. "enter", "tab", "escape", "backspace", "delete", "up", "down", "left", "right", "space", "win", "capslock").
    """
    if pyautogui is None:
        return "Error: pyautogui not available on this platform."
    try:
        await asyncio.to_thread(pyautogui.press, key.lower())
        return f"Pressed: {key}"
    except Exception as e:
        return f"Error pressing key: {e}"


_BLOCKED_HOTKEY_RULES = [
    ({"ctrl", "c"}, "Ctrl+C 会终止进程"),
    ({"ctrl", "break"}, "Ctrl+Break 会终止进程"),
    ({"alt", "f4"}, "Alt+F4 会关闭窗口"),
]


def _check_blocked_hotkey(key_list: list[str]) -> str | None:
    normalized = {k.strip().lower() for k in key_list}
    for blocked_set, reason in _BLOCKED_HOTKEY_RULES:
        if blocked_set.issubset(normalized):
            return reason
    return None


async def windows_hotkey(keys: str) -> str:
    """Press a keyboard shortcut / combination of keys held simultaneously.

    Args:
        keys: Comma-separated key names to press together. E.g. "ctrl,c" for Ctrl+C, "alt,tab" for Alt+Tab, "ctrl,shift,esc" for Ctrl+Shift+Esc.
    """
    if pyautogui is None:
        return "Error: pyautogui not available on this platform."
    try:
        if "+" in keys:
            key_list = [k.strip().lower() for k in keys.split("+")]
        else:
            key_list = [k.strip().lower() for k in keys.split(",")]

        logger.info(f"[HOTKEY] 原始输入: keys={keys!r}")
        logger.info(f"[HOTKEY] 解析后: key_list={key_list}")
        logger.info(f"[HOTKEY] 即将调用 pyautogui.hotkey({', '.join(repr(k) for k in key_list)})")

        blocked = _check_blocked_hotkey(key_list)
        if blocked:
            logger.warning(f"[HOTKEY] 被阻止: {blocked}")
            return f"Error: {blocked}，已屏蔽此组合键。"

        await asyncio.to_thread(pyautogui.hotkey, *key_list)
        logger.info(f"[HOTKEY] 执行完成: {'+'.join(key_list)}")
        return f"Hotkey: {'+'.join(key_list)}"
    except Exception as e:
        logger.error(f"[HOTKEY] 执行出错: {e}")
        return f"Error pressing hotkey: {e}"


def get_tools() -> list:
    from src.agent.tooldef import ToolDef

    return [
        ToolDef.from_function(windows_screenshot),
        ToolDef.from_function(windows_press),
        ToolDef.from_function(windows_hotkey),
    ]
