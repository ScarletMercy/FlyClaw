"""DNS-pinning HTTP transport that prevents DNS rebinding (TOCTOU) attacks.

``_PinnedDnsTransport`` wraps a standard ``httpx.AsyncHTTPTransport`` and
intercepts every outgoing request:

1. Extracts the hostname from the request URL.
2. Resolves DNS **once** and validates every returned IP against private /
   internal / cloud-metadata ranges (reusing logic from ``url_safety``).
3. Rewrites the URL host to the pinned IP so httpcore connects directly.
4. Preserves the original hostname via the ``Host`` header and the
   ``sni_hostname`` extension so TLS certificate validation still works.

This eliminates the window between "check" and "connect" that an attacker
could exploit by changing DNS records mid-flight.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging

import httpx

logger = logging.getLogger("flyclaw.security.pinned_transport")


def _resolve_and_pin(hostname: str) -> tuple[bool, str, str]:
    """Resolve *hostname*, validate IPs, return a pinned IP string.

    Returns ``(safe, reason, pinned_ip)``.  When *safe* is ``True``,
    *pinned_ip* is the first validated IP address the transport should
    connect to.

    Validation logic is delegated to :func:`resolve_safe_ips` so that
    there is a **single source of truth** for hostname/IP checks (blocked
    hostnames, cloud-metadata IPs, private ranges, etc.).
    """
    from src.security.url_safety import (
        _allow_private_urls,
        _is_always_blocked,
        _is_private_ip,
        resolve_safe_ips,
    )

    hostname_lower = hostname.strip().lower().rstrip(".")
    if not hostname_lower:
        return False, "empty hostname", ""

    # Literal IP: validate directly without DNS round-trip.
    try:
        ip = ipaddress.ip_address(hostname_lower)
        if _is_always_blocked(ip):
            return False, f"cloud metadata address: {hostname}", ""
        if not _allow_private_urls() and _is_private_ip(ip):
            return False, f"private/internal address: {hostname}", ""
        return True, "", hostname_lower
    except ValueError:
        pass  # Not a literal IP — delegate to resolve_safe_ips.

    # Domain name: use the canonical resolver (single source of truth).
    # resolve_safe_ips checks _BLOCKED_HOSTNAMES, cloud metadata IPs,
    # private ranges, and DNS failures.
    safe, reason, safe_ips = resolve_safe_ips(hostname_lower)
    if not safe:
        return False, reason, ""
    return True, "", safe_ips[0]


def _build_host_header_value(hostname: str, port: int | None, scheme: str) -> str:
    """Return the correct ``Host`` header value for a pinned request.

    Standard ports (80/HTTP, 443/HTTPS) are omitted; non-standard ports
    are included.
    """
    default_port = 443 if scheme == "https" else 80
    if port is not None and port != default_port:
        return f"{hostname}:{port}"
    return hostname


class _PinnedDnsTransport(httpx.AsyncBaseTransport):
    """Async HTTP transport that pins DNS resolution to pre-validated IPs.

    Usage::

        transport = _PinnedDnsTransport()
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            resp = await client.get("https://example.com/page")

    **Redirect handling**: This transport does *not* follow redirects.
    The caller (``safe_fetch``) handles redirect chaining manually so
    that each hop gets its own DNS resolution + validation + pinning.
    """

    def __init__(self, **kwargs: object) -> None:
        # Pass through any httpx.AsyncHTTPTransport kwargs (verify, timeout, etc.)
        self._inner = httpx.AsyncHTTPTransport(**kwargs)  # type: ignore[arg-type]

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # --- Phase 1: DNS resolution + validation ---
        hostname = request.url.host
        scheme = request.url.scheme

        safe, reason, pinned_ip = await asyncio.to_thread(_resolve_and_pin, hostname)
        if not safe:
            raise ValueError(f"Blocked: {reason}")

        logger.debug("DNS pin: %s -> %s", hostname, pinned_ip)

        # --- Phase 2: Rewrite request to use pinned IP ---
        # Build the pinned URL (host replaced with IP).
        pinned_url = request.url.copy_with(host=pinned_ip)

        # Build the Host header value (original hostname).
        host_value = _build_host_header_value(hostname, request.url.port, scheme)

        # Merge extensions: add sni_hostname for TLS.
        extensions = dict(request.extensions)
        extensions["sni_hostname"] = hostname

        # Reconstruct the request with the pinned URL.
        # We must read the body before creating a new request because
        # httpx.Request stream can only be consumed once.
        body = await self._read_body(request)

        pinned_request = httpx.Request(
            method=request.method,
            url=pinned_url,
            headers={**request.headers, "host": host_value},
            content=body,
            extensions=extensions,
        )

        # --- Phase 3: Delegate to inner transport ---
        return await self._inner.handle_async_request(pinned_request)

    @staticmethod
    async def _read_body(request: httpx.Request) -> bytes | None:
        """Read the full request body (for replay in the pinned request)."""
        if request.stream is None:
            return None
        chunks: list[bytes] = []
        async for chunk in request.stream:
            if isinstance(chunk, str):
                chunks.append(chunk.encode())
            else:
                chunks.append(chunk)
        return b"".join(chunks) if chunks else None

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def __aenter__(self) -> _PinnedDnsTransport:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()
