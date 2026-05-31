"""Shared FTS5 query sanitization utilities."""

import re


def sanitize_fts5_query(query: str) -> str:
    """Build a safe FTS5 MATCH query from raw user input.

    - Extracts Chinese words and ASCII alphanumeric tokens via regex.
    - Strips FTS5-special characters (``"`` ``*`` ``(`` ``)``) before tokenization.
    - Joins surviving tokens with ``OR`` for broad recall.
    - Returns ``'""'`` for empty / whitespace-only input.
    """
    query = query.strip()
    if not query:
        return '""'
    # Strip FTS5-special characters so they cannot interfere with tokenization
    query = query.replace('"', " ").replace("*", " ").replace("(", " ").replace(")", " ")
    tokens = re.findall(r"[一-鿿]+|[a-zA-Z0-9_]+", query)
    if not tokens:
        return '""'
    return " OR ".join(tokens)
