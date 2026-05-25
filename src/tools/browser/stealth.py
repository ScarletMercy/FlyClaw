"""Anti-detection setup for Playwright browser contexts."""

from __future__ import annotations

import logging

logger = logging.getLogger("flyclaw.browser.stealth")

_STEALTH_AVAILABLE = False
try:
    from playwright_stealth import stealth_async  # type: ignore[import-untyped]
    _STEALTH_AVAILABLE = True
except ImportError:
    stealth_async = None  # type: ignore[assignment]


async def apply_stealth(page) -> bool:
    """Apply stealth patches to a Playwright page.

    Returns True if stealth was applied, False if unavailable.
    """
    if not _STEALTH_AVAILABLE or stealth_async is None:
        logger.debug("playwright-stealth not available, skipping")
        return False
    try:
        await stealth_async(page)
        logger.debug("Stealth patches applied")
        return True
    except Exception as e:
        logger.warning("Failed to apply stealth patches: %s", e)
        return False
