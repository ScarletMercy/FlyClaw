"""Markdown chunking for the memory/RAG system.

Splits text into overlapping chunks at paragraph boundaries,
targeting a configurable token count per chunk.
"""

from __future__ import annotations

import re

_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def chunk_markdown(
    text: str,
    chunk_tokens: int = 400,
    overlap_tokens: int = 80,
) -> list[dict]:
    """Split markdown text into overlapping chunks.

    Returns list of {"text": str, "index": int, "start_char": int, "end_char": int}.

    Strategy: split on paragraph boundaries (double newline), accumulate
    until token estimate exceeds chunk_tokens, then start new chunk with
    overlap from the previous chunk.
    """
    if not text.strip():
        return []

    # Split into paragraphs
    paragraphs = re.split(r"\n\n+", text)
    paragraphs = [p for p in paragraphs if p.strip()]

    chunks: list[dict] = []
    current_parts: list[str] = []
    current_chars = 0
    overlap_chars = overlap_tokens * _CHARS_PER_TOKEN
    chunk_start = 0

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)
        new_chars = current_chars + len(para)

        if current_parts and _estimate_tokens("".join(current_parts) + para) > chunk_tokens:
            # Flush current chunk
            chunk_text = "\n\n".join(current_parts)
            chunks.append(
                {
                    "text": chunk_text,
                    "index": len(chunks),
                    "start_char": chunk_start,
                    "end_char": chunk_start + len(chunk_text),
                }
            )

            # Compute overlap from the end of flushed chunk
            overlap_text = _extract_overlap(chunk_text, overlap_chars)
            current_parts = [overlap_text, para] if overlap_text else [para]
            current_chars = sum(len(p) for p in current_parts)
            chunk_start = (
                chunk_start + len(chunk_text) - len(overlap_text) if overlap_text else chunk_start + len(chunk_text)
            )
        else:
            current_parts.append(para)
            current_chars = new_chars

    # Flush remaining
    if current_parts:
        chunk_text = "\n\n".join(current_parts)
        chunks.append(
            {
                "text": chunk_text,
                "index": len(chunks),
                "start_char": chunk_start,
                "end_char": chunk_start + len(chunk_text),
            }
        )

    return chunks


def _extract_overlap(text: str, max_chars: int) -> str:
    """Extract tail portion of text for overlap with next chunk."""
    if max_chars <= 0 or len(text) <= max_chars:
        return ""
    # Take from the last paragraph boundary within the overlap window
    window = text[-max_chars:]
    cut = window.find("\n\n")
    if 0 < cut < len(window) - 1:
        return window[cut + 2 :]
    return window
