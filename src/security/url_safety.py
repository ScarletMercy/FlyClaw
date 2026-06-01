"""URL safety checks — blocks requests to private/internal network addresses.

Prevents SSRF (Server-Side Request Forgery) where a malicious prompt could
trick the agent into fetching internal resources like cloud metadata endpoints
(169.254.169.254), localhost services, or private network hosts.

The check can be disabled via ``security.allow_private_urls: true`` in config
or ``FLYCLAW_ALLOW_PRIVATE_URLS=true`` env var. Even when disabled, cloud
metadata endpoints are **always** blocked.
"""

import ipaddress
import logging
import os
import socket
from urllib.parse import urlparse

import httpx

logger = logging.getLogger("flyclaw.security.url_safety")

# Cloud metadata hostnames — always blocked regardless of config
_BLOCKED_HOSTNAMES = frozenset(
    {
        "metadata.google.internal",
        "metadata.goog",
    }
)

# Cloud metadata IPs — always blocked even with allow_private_urls=true
_ALWAYS_BLOCKED_IPS = frozenset(
    {
        ipaddress.ip_address("169.254.169.254"),  # AWS/GCP/Azure/DO/Oracle
        ipaddress.ip_address("169.254.170.2"),  # AWS ECS task IAM creds
        ipaddress.ip_address("169.254.169.253"),  # Azure IMDS
        ipaddress.ip_address("fd00:ec2::254"),  # AWS metadata IPv6
        ipaddress.ip_address("100.100.100.200"),  # Alibaba Cloud
    }
)

_ALWAYS_BLOCKED_NETWORKS = (
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local (no legit agent target)
)

# 100.64.0.0/10 CGNAT — not covered by ipaddress.is_private
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")

_allow_private_cache: bool | None = None


def _allow_private_urls() -> bool:
    """Check if private URL resolution is allowed. Cached for process lifetime."""
    global _allow_private_cache
    if _allow_private_cache is not None:
        return _allow_private_cache

    # Env var override (highest priority)
    env = os.getenv("FLYCLAW_ALLOW_PRIVATE_URLS", "").strip().lower()
    if env in ("true", "1", "yes"):
        _allow_private_cache = True
        return True
    if env in ("false", "0", "no"):
        _allow_private_cache = False
        return False

    # Config file
    try:
        from src.config import load_config

        cfg = load_config()
        _allow_private_cache = getattr(cfg.security, "allow_private_urls", False)
    except Exception:
        _allow_private_cache = False

    return _allow_private_cache


def reset_cache():
    """Reset cached config — for tests."""
    global _allow_private_cache
    _allow_private_cache = None


def _is_private_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        return True
    if ip.is_multicast or ip.is_unspecified:
        return True
    if ip in _CGNAT_NETWORK:
        return True
    return False


def _is_always_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip in _ALWAYS_BLOCKED_IPS:
        return True
    return any(ip in net for net in _ALWAYS_BLOCKED_NETWORKS)


def resolve_safe_ips(hostname: str) -> tuple[bool, str, list[str]]:
    """Resolve hostname and validate all IPs atomically.

    Returns ``(safe, reason, safe_ips)`` where *safe_ips* is a list of
    IP address strings that passed validation.  Callers that need DNS
    pinning (e.g. ``_PinnedDnsTransport``) use the returned IPs to connect
    directly, eliminating the TOCTOU window that enables DNS rebinding.

    Fails closed: DNS errors, parse failures, and blocked IPs all reject.
    Cloud metadata endpoints are always blocked regardless of config.
    """
    hostname = hostname.strip().lower().rstrip(".")
    if not hostname:
        return False, "empty hostname", []

    # Always block known metadata hostnames
    if hostname in _BLOCKED_HOSTNAMES:
        return False, f"blocked hostname: {hostname}", []

    # Resolve hostname to IP(s)
    try:
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        return False, f"DNS resolution failed: {hostname}", []

    safe_ips: list[str] = []
    for _family, _, _, _, sockaddr in addr_info:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue

        # Always block cloud metadata IPs
        if _is_always_blocked(ip):
            return False, f"cloud metadata address: {hostname} -> {ip_str}", []

        # Check private IPs (unless allowed by config)
        if not _allow_private_urls() and _is_private_ip(ip):
            return False, f"private/internal address: {hostname} -> {ip_str}", []

        safe_ips.append(ip_str)

    if not safe_ips:
        return False, f"no valid IP addresses for {hostname}", []

    return True, "", safe_ips


def is_safe_url(url: str) -> tuple[bool, str]:
    """Check if a URL is safe to fetch. Returns (safe, reason).

    Fails closed: DNS errors and parse failures block the request.
    Cloud metadata endpoints are always blocked regardless of config.
    """
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").strip().lower().rstrip(".")
        if not hostname:
            return False, "empty hostname"

        safe, reason, _ = resolve_safe_ips(hostname)
        return safe, reason

    except Exception as exc:
        return False, f"URL safety check error: {exc}"


# ── DNS-pinned safe HTTP fetch ────────────────────────────────────

_REDIRECT_CODES = {301, 302, 303, 307, 308}

_DEFAULT_FETCH_HEADERS: dict[str, str] = {
    "User-Agent": "flyclaw/1.0",
    "Accept": "*/*",
}


async def safe_fetch(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    content: bytes | str | None = None,
    timeout: float = 30.0,
    max_redirects: int = 10,
    follow_redirects: bool = True,
) -> httpx.Response:
    """Fetch *url* with DNS-pinned SSRF protection.

    1. Validates the URL scheme (only ``http``/``https``).
    2. Creates an ``httpx.AsyncClient`` backed by
       :class:`~src.security.pinned_transport._PinnedDnsTransport`
       which resolves DNS **once** and pins the connection to a validated IP.
    3. If *follow_redirects* is ``True``, each redirect hop is resolved,
       validated, and pinned independently.

    Raises:
        ValueError: When the URL or any resolved IP is blocked.
        httpx.HTTPError: On network-level failures.
    """
    import httpx as _httpx
    from urllib.parse import urljoin

    from src.security.pinned_transport import _PinnedDnsTransport

    # Basic URL validation
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    if not parsed.hostname:
        raise ValueError("URL has no hostname")
    if len(url) > 2000:
        raise ValueError(f"URL exceeds maximum length of 2000 characters")

    merged_headers = {**_DEFAULT_FETCH_HEADERS, **(headers or {})}

    transport = _PinnedDnsTransport()
    async with _httpx.AsyncClient(
        transport=transport,
        follow_redirects=False,
        timeout=timeout,
    ) as client:
        current_url = url
        for _ in range(max_redirects + 1):
            # Build request body
            if isinstance(content, str):
                body: bytes | None = content.encode()
            else:
                body = content

            request = client.build_request(
                method=method,
                url=current_url,
                headers=merged_headers,
                content=body,
            )
            response = await client.send(request)

            if follow_redirects and response.status_code in _REDIRECT_CODES:
                location = response.headers.get("location")
                if not location:
                    raise ValueError(f"Redirect {response.status_code} without Location header")
                current_url = urljoin(current_url, location)
                # POST/PUT redirects become GET per HTTP convention
                if response.status_code in (301, 302, 303) and method in ("POST", "PUT", "PATCH"):
                    method = "GET"
                    body = None
                continue

            return response

    raise ValueError(f"Too many redirects (>{max_redirects})")
