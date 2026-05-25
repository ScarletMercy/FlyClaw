from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

logger = logging.getLogger("flyclaw.links")

URL_PATTERN = re.compile(r"https?://[^\s<>\)\]\"]+", re.IGNORECASE)


async def _fetch_preview(url: str, max_chars: int = 400) -> Optional[str]:
    """Fetch a brief preview for a URL using the web_fetch tool."""
    try:
        from src.tools.web_tools import web_fetch

        result = await web_fetch(url)
        if isinstance(result, str) and len(result) > max_chars:
            result = result[:max_chars]
        if (
            isinstance(result, str)
            and result
            and not result.startswith("[error]")
            and not result.startswith("[web_fetch")
        ):
            # Extract first meaningful line as title
            lines = result.strip().split("\n")
            title = ""
            body_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped:
                    if not title:
                        title = stripped[:150]
                    else:
                        body_lines.append(stripped)
                        if len("\n".join(body_lines)) > 300:
                            break

            body = " ".join(body_lines)[:300]
            if title:
                preview = f"📎 **{title}**"
                if body:
                    preview += f"\n   {body}"
                return preview
        return None
    except Exception as e:
        logger.debug("Link preview failed for %s: %s", url, e)
        return None


async def detect_and_preview_links(text: str, max_previews: int = 3) -> str:
    """Detect URLs in text and generate previews.

    Args:
        text: Message text to scan for URLs
        max_previews: Maximum number of link previews to generate

    Returns:
        Formatted preview string, or empty string if no links found
    """
    urls = URL_PATTERN.findall(text)
    if not urls:
        return ""

    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        clean = url.rstrip(".,;:!?)>")
        if clean not in seen:
            seen.add(clean)
            unique_urls.append(clean)

    unique_urls = unique_urls[:max_previews]

    previews = []
    for url in unique_urls:
        preview = await _fetch_preview(url)
        if preview:
            previews.append(f"{preview}\n   🔗 {url}")

    if not previews:
        return ""

    return "\n\n---\n**Link Preview**\n" + "\n".join(previews)
