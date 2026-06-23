"""OpenAI message content → plain text extraction.

Single source of truth for flattening `content` (str or content-blocks list)
into text. Media blocks (image_url / video_url, which carry base64) are
deliberately skipped so base64 never leaks into summaries, the session index,
or prompt assembly. Callers wanting the legacy bare-string-in-list behavior
pass `allow_bare_str=True`; callers needing a different separator pass `joiner`.
"""

from __future__ import annotations


def content_to_text(content, *, joiner: str = "\n", allow_bare_str: bool = False) -> str:
    """Flatten OpenAI message `content` to plain text.

    - str content is returned as-is.
    - list content: only ``{"type": "text", "text": ...}`` blocks are joined
      with ``joiner`` (empty text blocks dropped). Media blocks
      (image_url / video_url / etc.) are skipped — their base64 must never
      enter downstream text paths. With ``allow_bare_str=True``, bare strings
      inside the list are joined too.
    - any other type falls back to ``""``.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text", "")
                if t:
                    parts.append(t)
            elif allow_bare_str and isinstance(block, str) and block:
                parts.append(block)
        return joiner.join(parts)
    return ""
