"""Browser session manager — lifecycle, session isolation, cleanup."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("myclaw.browser.manager")

# ── Module-level singleton — delegates to ServiceContainer ──

from src._container import get_container


def get_browser_manager() -> BrowserManager:
    return get_container().browser_manager


@dataclass
class BrowserSession:
    browser: object  # playwright Browser
    context: object  # playwright BrowserContext
    page: object  # playwright Page
    last_used: float = field(default_factory=time.time)


class BrowserManager:
    """Manages Playwright browser instances with per-session isolation."""

    def __init__(self):
        self._sessions: dict[str, BrowserSession] = {}
        self._config = None
        self._playwright = None
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    def _load_config(self):
        if self._config is None:
            from src.config import load_config
            self._config = load_config().tools.browser
        return self._config

    async def _ensure_playwright(self):
        if self._playwright is not None:
            return self._playwright
        from playwright.async_api import async_playwright
        self._playwright = await async_playwright().start()
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._idle_cleanup_loop())
        return self._playwright

    def _find_browser_executable(self, browser_name: str) -> str | None:
        """Find an installed Playwright browser binary, any revision."""
        import glob
        import sys

        if sys.platform == "win32":
            _default_base = os.path.join(
                os.environ.get("LOCALAPPDATA", ""), "ms-playwright",
            )
        elif sys.platform == "darwin":
            _default_base = os.path.expanduser("~/Library/Caches/ms-playwright")
        else:
            _default_base = os.path.expanduser("~/.cache/ms-playwright")

        base = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", _default_base)

        if browser_name == "chromium":
            if sys.platform == "win32":
                pattern = os.path.join(base, "chromium-*", "chrome-win64", "chrome.exe")
            elif sys.platform == "darwin":
                pattern = os.path.join(base, "chromium-*", "chrome-mac", "Chromium.app", "Contents", "MacOS", "Chromium")
            else:
                pattern = os.path.join(base, "chromium-*", "chrome-linux", "chrome")
        elif browser_name == "firefox":
            if sys.platform == "win32":
                pattern = os.path.join(base, "firefox-*", "**", "firefox.exe")
            elif sys.platform == "darwin":
                pattern = os.path.join(base, "firefox-*", "**", "firefox")
            else:
                pattern = os.path.join(base, "firefox-*", "**", "firefox")
        elif browser_name == "webkit":
            if sys.platform == "win32":
                pattern = os.path.join(base, "webkit-*", "**", "Playwright.exe")
            elif sys.platform == "darwin":
                pattern = os.path.join(base, "webkit-*", "**", "minibrowser-wpe")
            else:
                pattern = os.path.join(base, "webkit-*", "**", "minibrowser-wpe")
        else:
            return None
        matches = sorted(glob.glob(pattern, recursive=True), reverse=True)
        return matches[0] if matches else None

    async def get_page(self, session_id: str = "default"):
        """Get or create a browser page for the given session."""
        config = self._load_config()

        async with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.last_used = time.time()
                return session.page

            if len(self._sessions) >= config.max_sessions:
                oldest_id = min(self._sessions, key=lambda k: self._sessions[k].last_used)
                await self._close_session(oldest_id)

            pw = await self._ensure_playwright()

            if config.cdp_url:
                browser = await pw.chromium.connect_over_cdp(config.cdp_url)
                context = browser.contexts[0] if browser.contexts else await browser.new_context(
                    viewport={"width": config.viewport_width, "height": config.viewport_height},
                )
            else:
                launch_args = []
                if config.block_urls:
                    pass  # Handled at context level

                # Auto-detect installed browser binary
                exec_path = self._find_browser_executable(config.browser)

                launch_kwargs = {
                    "headless": config.headless,
                    "args": launch_args,
                }
                if exec_path:
                    launch_kwargs["executable_path"] = exec_path
                    logger.info("Using browser at: %s", exec_path)

                if config.user_data_dir:
                    context = await pw.chromium.launch_persistent_context(
                        user_data_dir=config.user_data_dir,
                        **launch_kwargs,
                        viewport={"width": config.viewport_width, "height": config.viewport_height},
                    )
                    browser = context.browser
                else:
                    browser_type = getattr(pw, config.browser)
                    browser = await browser_type.launch(**launch_kwargs)
                    context_kwargs = {
                        "viewport": {"width": config.viewport_width, "height": config.viewport_height},
                    }
                    if config.block_urls:
                        context_kwargs["block_resource_urls"] = config.block_urls
                    context = await browser.new_context(**context_kwargs)

            page = await context.new_page()
            page.set_default_timeout(config.timeout_seconds * 1000)

            if config.stealth:
                from src.tools.browser.stealth import apply_stealth
                await apply_stealth(page)

            self._sessions[session_id] = BrowserSession(
                browser=browser,
                context=context,
                page=page,
                last_used=time.time(),
            )
            logger.info("Browser session created: %s (%s)", session_id,
                        "CDP" if config.cdp_url else config.browser)
            return page

    async def _close_session(self, session_id: str):
        if session_id not in self._sessions:
            return
        session = self._sessions.pop(session_id)
        try:
            await session.context.close()
        except Exception:
            pass
        try:
            if not session.browser._was_closed:  # type: ignore
                await session.browser.close()
        except Exception:
            pass
        logger.info("Browser session closed: %s", session_id)

    async def close_session(self, session_id: str = "default"):
        async with self._lock:
            await self._close_session(session_id)

    async def close_all(self):
        async with self._lock:
            for sid in list(self._sessions):
                await self._close_session(sid)
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    async def _idle_cleanup_loop(self):
        """Background task: close sessions idle > 10 minutes."""
        while True:
            await asyncio.sleep(60)
            try:
                config = self._load_config()
                idle_limit = 600  # 10 minutes
                now = time.time()
                async with self._lock:
                    for sid in list(self._sessions):
                        if now - self._sessions[sid].last_used > idle_limit:
                            logger.info("Closing idle browser session: %s", sid)
                            await self._close_session(sid)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning("Idle cleanup error: %s", e)
