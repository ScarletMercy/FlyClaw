"""Credential redaction — strip API keys and secrets from text.

Applies regex pattern matching to mask API keys, tokens, passwords,
and other credentials before they reach model outputs or logs.

Markdown code regions (fenced blocks and inline code) are protected
from modification so that model-generated examples and syntax are
preserved intact.
"""

from __future__ import annotations

import re

from src.security.credential_patterns import CREDENTIAL_PATTERNS

_PREFIX_PATTERNS = [cp.pattern for cp in CREDENTIAL_PATTERNS]

_PREFIX_RE = re.compile(r"(?<![A-Za-z0-9_-])(" + "|".join(_PREFIX_PATTERNS) + r")(?![A-Za-z0-9_-])")

# ---------------------------------------------------------------------------
# Markdown code-region protection
# ---------------------------------------------------------------------------

# Null byte used as placeholder delimiter — excluded from all greedy groups.
NULL = "\0"

# Fenced code blocks: ```lang\n...\n```  (CommonMark: closing must be own line)
_MD_FENCE_BLOCK_RE = re.compile(
    r"(^[ \t]*`{3,}[^\n]*\n)([\s\S]*?)(^[ \t]*`{3,}\s*$)",
    re.MULTILINE,
)

# Inline code: `code`  — negative lookbehind/lookahead avoids triple-backtick
_MD_INLINE_RE = re.compile(r"(?<!`)(`[^`\n]+`)(?!`)")


def _protect_markdown(text: str) -> tuple[str, dict[str, str]]:
    """Replace markdown code blocks and inline code with placeholders.

    Credentials inside code regions are still redacted — only the structural
    delimiters (backticks, fences) are preserved so the markdown remains valid.
    """
    placeholders: dict[str, str] = {}
    idx = 0

    def _sub_fence(m: re.Match) -> str:
        nonlocal idx
        key = f"{NULL}MKD{idx}{NULL}"
        idx += 1
        # Redact credentials inside the code block body, keep fence delimiters
        opener, body, closer = m.group(1), m.group(2), m.group(3)
        placeholders[key] = opener + _redact_credentials_only(body) + closer
        return key

    def _sub_inline(m: re.Match) -> str:
        nonlocal idx
        key = f"{NULL}MKD{idx}{NULL}"
        idx += 1
        # Redact credentials inside inline code, keep backtick delimiters
        full = m.group(0)
        placeholders[key] = full[0] + _redact_credentials_only(full[1:-1]) + full[-1]
        return key

    text = _MD_FENCE_BLOCK_RE.sub(_sub_fence, text)
    text = _MD_INLINE_RE.sub(_sub_inline, text)
    return text, placeholders


def _restore_markdown(text: str, placeholders: dict[str, str]) -> str:
    """Restore markdown code blocks and inline code from placeholders."""
    for key, value in placeholders.items():
        text = text.replace(key, value)
    return text


def _redact_credentials_only(text: str) -> str:
    """Apply credential redaction patterns to text without markdown protection.

    Used for code-block bodies where credentials should still be masked
    but the surrounding fence structure must be preserved.
    """
    if not text or not isinstance(text, str):
        return text

    text = _PREFIX_RE.sub(lambda m: _mask(m.group(1)), text)

    def _redact_env_quoted(m):
        name, quote, value = m.group(1), m.group(2), m.group(3)
        return f"{name}={quote}{_mask(value)}{quote}"

    text = _ENV_QUOTED_RE.sub(_redact_env_quoted, text)

    def _redact_env_unquoted(m):
        name, value = m.group(1), m.group(2)
        return f"{name}={_mask(value)}"

    text = _ENV_UNQUOTED_RE.sub(_redact_env_unquoted, text)

    def _redact_json(m):
        key, value = m.group(1), m.group(2)
        return f'{key}: "{_mask(value)}"'

    text = _JSON_FIELD_RE.sub(_redact_json, text)
    text = _AUTH_HEADER_RE.sub(
        lambda m: m.group(1) + _mask(m.group(2)),
        text,
    )
    text = _PRIVATE_KEY_RE.sub("[REDACTED PRIVATE KEY]", text)
    text = _DB_CONNSTR_RE.sub(lambda m: f"{m.group(1)}***{m.group(3)}", text)
    text = _JWT_RE.sub(lambda m: _mask(m.group(0)), text)
    text = _URL_USERINFO_RE.sub(
        lambda m: f"{m.group(1)}://{m.group(2)}:***@",
        text,
    )

    def _redact_url_query(m):
        scheme, authority, path, query, fragment = (
            m.group(1),
            m.group(2),
            m.group(3),
            m.group(4),
            m.group(5) or "",
        )
        return f"{scheme}://{authority}{path}?{_redact_query_string(query)}{fragment}"

    text = _URL_WITH_QUERY_RE.sub(_redact_url_query, text)
    return text


# ---------------------------------------------------------------------------
# Credential detection patterns
# ---------------------------------------------------------------------------

# ENV assignments: API_KEY=value
_SECRET_ENV_NAMES = r"(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)"

# Quoted: API_KEY="value"  —  \2 backreference provides natural boundary
_ENV_QUOTED_RE = re.compile(
    rf'([A-Z0-9_]{{0,50}}{_SECRET_ENV_NAMES}[A-Z0-9_]{{0,50}})\s*=(["\x27])([^{NULL}]+?)\2',
)

# Unquoted: API_KEY=value  —  stop at whitespace, ), backtick, quotes, NULL
_ENV_UNQUOTED_RE = re.compile(
    rf'([A-Z0-9_]{{0,50}}{_SECRET_ENV_NAMES}[A-Z0-9_]{{0,50}})\s*=([^\s)`\x27"{NULL}]+)',
)

# JSON fields: "apiKey": "value"
_JSON_KEY_NAMES = (
    r"(?:api_?[Kk]ey|token|secret|password|access_token|refresh_token|auth_token|bearer|secret_value|private_key)"
)
_JSON_FIELD_RE = re.compile(
    r'("' + _JSON_KEY_NAMES + r'")\s*:\s*"([^"' + NULL + r']+)"',
    re.IGNORECASE,
)

# Authorization headers — stop at ), backtick, NULL
_AUTH_HEADER_RE = re.compile(
    r"(Authorization:\s*Bearer\s+)([^\s)`" + NULL + r"]+)",
    re.IGNORECASE,
)

# Private key blocks
_PRIVATE_KEY_RE = re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----")

# DB connection strings: protocol://user:PASSWORD@host
_DB_CONNSTR_RE = re.compile(
    r"((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:\s" + NULL + r"]+:)([^@" + NULL + r"]+)(@)",
    re.IGNORECASE,
)

# JWT tokens: eyJ... (base64-encoded JSON) — require 3 dot-separated segments
_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{4,}")

# URL userinfo: scheme://user:password@host (non-DB) — exclude NULL from user/pass
_URL_USERINFO_RE = re.compile(
    r"(https?|wss?|ftp)://([^/\s:@" + NULL + r"]+):([^/\s@" + NULL + r"]+)@",
)

# URL query params with sensitive names — exclude ), `, NULL from query and fragment
_SENSITIVE_QUERY_PARAMS = frozenset(
    {
        "access_token",
        "refresh_token",
        "id_token",
        "token",
        "api_key",
        "apikey",
        "client_secret",
        "password",
        "auth",
        "jwt",
        "secret",
        "key",
        "code",
        "signature",
    }
)
_URL_WITH_QUERY_RE = re.compile(
    r"(https?|wss?|ftp)://([^\s/?#"
    + NULL
    + r"]+)([^\s?#"
    + NULL
    + r"]*)\?([^\s#)`"
    + NULL
    + r"]+)(#[^)\s`"
    + NULL
    + r"]*)?",
)


def _mask(token: str) -> str:
    """Mask a secret, preserving first 6 and last 4 chars if long enough."""
    if not token:
        return "***"
    if len(token) < 18:
        return "***"
    return f"{token[:6]}...{token[-4:]}"


def _redact_query_string(query: str) -> str:
    if not query:
        return query
    parts = []
    for pair in query.split("&"):
        if "=" not in pair:
            parts.append(pair)
            continue
        key, _, value = pair.partition("=")
        if key.lower() in _SENSITIVE_QUERY_PARAMS:
            parts.append(f"{key}=***")
        else:
            parts.append(pair)
    return "&".join(parts)


def redact(text: str) -> str:
    """Apply all redaction patterns to text.

    Safe to call on any string — non-matching text passes through unchanged.
    Markdown code blocks and inline code are protected from modification.
    Credentials inside fenced code blocks are still redacted.
    """
    if not text or not isinstance(text, str):
        return text

    # --- Protect markdown code regions from redaction ---
    text, placeholders = _protect_markdown(text)

    # Apply credential redaction to non-code regions
    text = _redact_credentials_only(text)

    # --- Restore markdown code regions ---
    text = _restore_markdown(text, placeholders)
    return text
