"""Browser automation tools — navigate, click, type, scroll, snapshot, etc."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextvars import ContextVar
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger("myclaw.browser.tools")

_current_session: ContextVar[str] = ContextVar("_browser_session", default="default")


def set_browser_session(session_id: str):
    _current_session.set(session_id)


def _session_id() -> str:
    return _current_session.get("default")


@tool
async def browser_navigate(url: str) -> str:
    """Navigate browser to a URL. Returns page title and accessibility snapshot with element refs (@e1, @e2...).

    Args:
        url: The URL to navigate to
    """
    from src.tools.browser.manager import get_browser_manager
    from src.tools.browser.snapshot import get_snapshot, build_page_info

    try:
        mgr = get_browser_manager()
        page = await mgr.get_page(_session_id())
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_load_state("networkidle", timeout=10000)
    except Exception as e:
        return f"Error navigating to {url}: {e}"

    snap = await get_snapshot(page)
    return build_page_info(snap["url"], snap["title"], snap["snapshot"], snap["element_count"])


@tool
async def browser_snapshot(full: bool = False) -> str:
    """Get a text snapshot of the current page's accessibility tree. Elements have refs like @e1, @e2.

    Args:
        full: If True, include all elements. If False (default), compact view with interactive elements.
    """
    from src.tools.browser.manager import get_browser_manager
    from src.tools.browser.snapshot import get_snapshot, build_page_info

    try:
        mgr = get_browser_manager()
        page = await mgr.get_page(_session_id())
    except Exception as e:
        return f"Error: no browser session. Use browser_navigate first. ({e})"

    snap = await get_snapshot(page, compact=not full)
    return build_page_info(snap["url"], snap["title"], snap["snapshot"], snap["element_count"])


@tool
async def browser_click(ref: str) -> str:
    """Click an element on the page by its ref (e.g. @e1, @e2 from snapshot).

    Args:
        ref: Element reference from snapshot (e.g. "@e1")
    """
    from src.tools.browser.manager import get_browser_manager
    from src.tools.browser.snapshot import get_snapshot, build_page_info

    try:
        mgr = get_browser_manager()
        page = await mgr.get_page(_session_id())
    except Exception as e:
        return f"Error: no browser session. ({e})"

    locator = await _ref_to_locator(page, ref)
    if not locator:
        return f"Error: element {ref} not found. Take a new snapshot with browser_snapshot."

    try:
        await locator.click(timeout=10000)
        await page.wait_for_load_state("networkidle", timeout=5000)
    except Exception as e:
        return f"Error clicking {ref}: {e}"

    snap = await get_snapshot(page)
    return f"Clicked {ref}\n{build_page_info(snap['url'], snap['title'], snap['snapshot'], snap['element_count'])}"


@tool
async def browser_type(ref: str, text: str, submit: bool = False) -> str:
    """Type text into an input element identified by ref. Optionally press Enter after typing.

    Args:
        ref: Element reference from snapshot (e.g. "@e1")
        text: Text to type
        submit: If True, press Enter after typing (to submit form)
    """
    from src.tools.browser.manager import get_browser_manager
    from src.tools.browser.snapshot import get_snapshot, build_page_info

    try:
        mgr = get_browser_manager()
        page = await mgr.get_page(_session_id())
    except Exception as e:
        return f"Error: no browser session. ({e})"

    locator = await _ref_to_locator(page, ref)
    if not locator:
        return f"Error: element {ref} not found. Take a new snapshot with browser_snapshot."

    try:
        await locator.click()
        await locator.fill(text)
        if submit:
            await locator.press("Enter")
            await page.wait_for_load_state("networkidle", timeout=5000)
    except Exception as e:
        return f"Error typing into {ref}: {e}"

    if submit:
        snap = await get_snapshot(page)
        return f"Typed '{text}' and submitted in {ref}\n{build_page_info(snap['url'], snap['title'], snap['snapshot'], snap['element_count'])}"
    return f"Typed '{text}' into {ref}"


@tool
async def browser_scroll(direction: str = "down", amount: int = 3) -> str:
    """Scroll the page in a direction.

    Args:
        direction: Scroll direction: "up", "down", "left", "right" (default "down")
        amount: Number of scroll steps (default 3, each ~300px)
    """
    from src.tools.browser.manager import get_browser_manager
    from src.tools.browser.snapshot import get_snapshot, build_page_info

    try:
        mgr = get_browser_manager()
        page = await mgr.get_page(_session_id())
    except Exception as e:
        return f"Error: no browser session. ({e})"

    delta_map = {
        "down": (0, amount * 300),
        "up": (0, -amount * 300),
        "right": (amount * 300, 0),
        "left": (-amount * 300, 0),
    }
    dx, dy = delta_map.get(direction, (0, amount * 300))

    try:
        await page.evaluate(f"window.scrollBy({dx}, {dy})")
    except Exception as e:
        return f"Error scrolling: {e}"

    snap = await get_snapshot(page)
    return f"Scrolled {direction} ({amount}x)\n{build_page_info(snap['url'], snap['title'], snap['snapshot'], snap['element_count'])}"


@tool
async def browser_back() -> str:
    """Go back in browser history. Returns snapshot of the previous page."""
    from src.tools.browser.manager import get_browser_manager
    from src.tools.browser.snapshot import get_snapshot, build_page_info

    try:
        mgr = get_browser_manager()
        page = await mgr.get_page(_session_id())
        await page.go_back(wait_until="domcontentloaded", timeout=15000)
        await page.wait_for_load_state("networkidle", timeout=5000)
    except Exception as e:
        return f"Error going back: {e}"

    snap = await get_snapshot(page)
    return build_page_info(snap["url"], snap["title"], snap["snapshot"], snap["element_count"])


@tool
async def browser_press(key: str) -> str:
    """Press a keyboard key (Enter, Tab, Escape, ArrowDown, etc.).

    Args:
        key: Key name (e.g. "Enter", "Tab", "Escape", "ArrowDown", "Control+a")
    """
    from src.tools.browser.manager import get_browser_manager
    from src.tools.browser.snapshot import get_snapshot, build_page_info

    try:
        mgr = get_browser_manager()
        page = await mgr.get_page(_session_id())
        await page.keyboard.press(key)
        await page.wait_for_load_state("networkidle", timeout=3000)
    except Exception as e:
        return f"Error pressing {key}: {e}"

    snap = await get_snapshot(page)
    return f"Pressed {key}\n{build_page_info(snap['url'], snap['title'], snap['snapshot'], snap['element_count'])}"


@tool
async def browser_screenshot(path: str = "") -> str:
    """Take a screenshot of the current page. Returns the file path.

    Args:
        path: Optional file path to save screenshot. Default: auto-generated in workspace.
    """
    from src.tools.browser.manager import get_browser_manager

    try:
        mgr = get_browser_manager()
        page = await mgr.get_page(_session_id())
    except Exception as e:
        return f"Error: no browser session. ({e})"

    if not path:
        workspace = os.path.expanduser("~/.myclaw/workspace")
        os.makedirs(workspace, exist_ok=True)
        path = os.path.join(workspace, f"screenshot_{int(time.time())}.png")

    try:
        await page.screenshot(path=path, full_page=False)
    except Exception as e:
        return f"Error taking screenshot: {e}"

    return f"Screenshot saved: {path}"


@tool
async def browser_console(expression: str) -> str:
    """Execute JavaScript in the browser console and return the result.

    Args:
        expression: JavaScript expression to evaluate
    """
    from src.tools.browser.manager import get_browser_manager

    try:
        mgr = get_browser_manager()
        page = await mgr.get_page(_session_id())
        result = await page.evaluate(expression)
    except Exception as e:
        return f"Error evaluating expression: {e}"

    if result is None:
        return "undefined"
    return str(result)


@tool
async def browser_close() -> str:
    """Close the current browser session and release resources."""
    from src.tools.browser.manager import get_browser_manager

    try:
        mgr = get_browser_manager()
        await mgr.close_session(_session_id())
        return "Browser session closed"
    except Exception as e:
        return f"Error closing browser: {e}"


async def _ref_to_locator(page, ref: str):
    """Convert an @eN ref to a Playwright locator.

    Uses CDP AX tree to find the element, then locates it by role/name.
    """
    clean = ref.lstrip("@")
    if not clean.startswith("e"):
        return None
    try:
        idx = int(clean[1:])
    except ValueError:
        return None

    try:
        cdp = getattr(page, "_ax_cdp_session", None)
        if cdp is None or getattr(cdp, "_was_closed", False):
            cdp = await page.context.new_cdp_session(page)
            page._ax_cdp_session = cdp
        result = await cdp.send("Accessibility.getFullAXTree")
    except Exception:
        return None

    from src.tools.browser.snapshot import get_interactive_nodes
    nodes = result.get("nodes", [])
    interactive = get_interactive_nodes(nodes)
    if idx < 1 or idx > len(interactive):
        return None

    node = interactive[idx - 1]
    role_obj = node.get("role", {})
    role = role_obj.get("value", "") if isinstance(role_obj, dict) else str(role_obj)
    name_obj = node.get("name", {})
    name = name_obj.get("value", "") if isinstance(name_obj, dict) else str(name_obj)

    if not role:
        return None
    try:
        if name:
            return page.get_by_role(role, name=name, exact=False)
        return page.get_by_role(role)
    except Exception:
        return None

