"""Credential redaction — strip API keys and secrets from text.

Applies regex pattern matching to mask API keys, tokens, passwords,
and other credentials before they reach model outputs or logs.
"""

from __future__ import annotations

import re

from src.security.credential_patterns import CREDENTIAL_PATTERNS

_PREFIX_PATTERNS = [cp.pattern for cp in CREDENTIAL_PATTERNS]

_PREFIX_RE = re.compile(r"(?<![A-Za-z0-9_-])(" + "|".join(_PREFIX_PATTERNS) + r")(?![A-Za-z0-9_-])")

# ENV assignments: API_KEY=value
_SECRET_ENV_NAMES = r"(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)"
_ENV_ASSIGN_RE = re.compile(
    rf"([A-Z0-9_]{{0,50}}{_SECRET_ENV_NAMES}[A-Z0-9_]{{0,50}})\s*=\s*(['\"]?)(\S+)\2",
)

# JSON fields: "apiKey": "value"
_JSON_KEY_NAMES = (
    r"(?:api_?[Kk]ey|token|secret|password|access_token|refresh_token|auth_token|bearer|secret_value|private_key)"
)
_JSON_FIELD_RE = re.compile(
    rf'("{_JSON_KEY_NAMES}")\s*:\s*"([^"]+)"',
    re.IGNORECASE,
)

# Authorization headers
_AUTH_HEADER_RE = re.compile(
    r"(Authorization:\s*Bearer\s+)(\S+)",
    re.IGNORECASE,
)

# Private key blocks
_PRIVATE_KEY_RE = re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----")

# DB connection strings: protocol://user:PASSWORD@host
_DB_CONNSTR_RE = re.compile(
    r"((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:]+:)([^@]+)(@)",
    re.IGNORECASE,
)

# JWT tokens: eyJ... (base64-encoded JSON) — require 3 dot-separated segments
_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{4,}")

# URL userinfo: scheme://user:password@host (non-DB)
_URL_USERINFO_RE = re.compile(
    r"(https?|wss?|ftp)://([^/\s:@]+):([^/\s@]+)@",
)

# URL query params with sensitive names
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
    r"(https?|wss?|ftp)://([^\s/?#]+)([^\s?#]*)\?([^\s#]+)(#\S*)?",
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
    """
    if not text or not isinstance(text, str):
        return text

    # Known API key prefixes
    text = _PREFIX_RE.sub(lambda m: _mask(m.group(1)), text)

    # ENV assignments
    def _redact_env(m):
        name, quote, value = m.group(1), m.group(2), m.group(3)
        return f"{name}={quote}{_mask(value)}{quote}"

    text = _ENV_ASSIGN_RE.sub(_redact_env, text)

    # JSON fields
    def _redact_json(m):
        key, value = m.group(1), m.group(2)
        return f'{key}: "{_mask(value)}"'

    text = _JSON_FIELD_RE.sub(_redact_json, text)

    # Authorization headers
    text = _AUTH_HEADER_RE.sub(
        lambda m: m.group(1) + _mask(m.group(2)),
        text,
    )

    # Private key blocks
    text = _PRIVATE_KEY_RE.sub("[REDACTED PRIVATE KEY]", text)

    # DB connection string passwords
    text = _DB_CONNSTR_RE.sub(lambda m: f"{m.group(1)}***{m.group(3)}", text)

    # JWT tokens
    text = _JWT_RE.sub(lambda m: _mask(m.group(0)), text)

    # URL userinfo
    text = _URL_USERINFO_RE.sub(
        lambda m: f"{m.group(1)}://{m.group(2)}:***@",
        text,
    )

    # URL query params
    def _redact_url_query(m):
        scheme, authority, path, query, fragment = (m.group(1), m.group(2), m.group(3), m.group(4), m.group(5) or "")
        return f"{scheme}://{authority}{path}?{_redact_query_string(query)}{fragment}"

    text = _URL_WITH_QUERY_RE.sub(_redact_url_query, text)

    return text
