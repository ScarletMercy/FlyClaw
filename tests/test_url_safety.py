"""Tests for DNS-pinned SSRF protection (url_safety + pinned_transport + safe_fetch).

Covers:
- resolve_safe_ips: public IPs, private IPs, DNS failures, always-blocked IPs
- _PinnedDnsTransport: DNS pinning, blocking, literal IP passthrough
- safe_fetch: redirect safety, blocked targets, HTTPS SNI preservation
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.security.url_safety import (
    is_safe_url,
    resolve_safe_ips,
    reset_cache,
    safe_fetch,
)


# ── Helpers ────────────────────────────────────────────────────────


def _mock_getaddrinfo(ips: list[str]):
    """Return a mock for socket.getaddrinfo that resolves to the given IPs."""

    def _resolve(host, port, family, type_, proto=0, flags=0):
        results = []
        for ip in ips:
            try:
                addr = ipaddress.ip_address(ip)
            except ValueError:
                continue
            family = socket.AF_INET if addr.version == 4 else socket.AF_INET6
            results.append((family, type_, proto, "", (ip, port or 0)))
        return results

    return _resolve


# ── resolve_safe_ips ───────────────────────────────────────────────


class TestResolveSafeIps:
    def setup_method(self):
        reset_cache()

    @patch("src.security.url_safety.socket.getaddrinfo")
    def test_public_ip_is_safe(self, mock_dns):
        mock_dns.return_value = _mock_getaddrinfo(["93.184.216.34"])(
            "example.com", None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
        safe, reason, ips = resolve_safe_ips("example.com")
        assert safe is True
        assert reason == ""
        assert "93.184.216.34" in ips

    @patch("src.security.url_safety.socket.getaddrinfo")
    def test_private_ip_blocked(self, mock_dns):
        mock_dns.return_value = _mock_getaddrinfo(["127.0.0.1"])("evil.com", None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        safe, reason, ips = resolve_safe_ips("evil.com")
        assert safe is False
        assert "private" in reason
        assert ips == []

    @patch("src.security.url_safety.socket.getaddrinfo")
    def test_cloud_metadata_blocked(self, mock_dns):
        mock_dns.return_value = _mock_getaddrinfo(["169.254.169.254"])(
            "evil.com", None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
        safe, reason, ips = resolve_safe_ips("evil.com")
        assert safe is False
        assert "cloud metadata" in reason

    @patch("src.security.url_safety.socket.getaddrinfo")
    def test_mixed_ips_one_private_blocks(self, mock_dns):
        """If any resolved IP is private, the entire resolution fails."""
        mock_dns.return_value = _mock_getaddrinfo(["93.184.216.34", "10.0.0.1"])(
            "evil.com", None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
        safe, reason, ips = resolve_safe_ips("evil.com")
        assert safe is False
        assert "private" in reason

    def test_dns_failure_blocks(self):
        safe, reason, ips = resolve_safe_ips("this.domain.does.not.exist.invalid")
        assert safe is False
        assert ips == []

    def test_empty_hostname(self):
        safe, reason, ips = resolve_safe_ips("")
        assert safe is False
        assert "empty" in reason

    @patch("src.security.url_safety.socket.getaddrinfo")
    def test_metadata_hostname_blocked(self, mock_dns):
        """metadata.google.internal is blocked by name, no DNS needed."""
        safe, reason, ips = resolve_safe_ips("metadata.google.internal")
        assert safe is False
        assert "blocked hostname" in reason

    @patch("src.security.url_safety.socket.getaddrinfo")
    def test_multiple_safe_ips_returned(self, mock_dns):
        mock_dns.return_value = _mock_getaddrinfo(["93.184.216.34", "93.184.216.35"])(
            "cdn.example.com", None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
        safe, reason, ips = resolve_safe_ips("cdn.example.com")
        assert safe is True
        assert len(ips) == 2


# ── is_safe_url (backward compat wrapper) ──────────────────────────


class TestIsSafeUrl:
    def setup_method(self):
        reset_cache()

    @patch("src.security.url_safety.socket.getaddrinfo")
    def test_delegates_to_resolve_safe_ips(self, mock_dns):
        mock_dns.return_value = _mock_getaddrinfo(["93.184.216.34"])(
            "example.com", None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
        safe, reason = is_safe_url("https://example.com/path")
        assert safe is True
        assert reason == ""

    def test_invalid_url(self):
        safe, reason = is_safe_url("not a url")
        assert safe is False


# ── _PinnedDnsTransport ────────────────────────────────────────────


class TestPinnedDnsTransport:
    def setup_method(self):
        reset_cache()

    @pytest.mark.asyncio
    @patch("src.security.url_safety.socket.getaddrinfo")
    async def test_pins_to_resolved_ip(self, mock_dns):
        """The transport should connect to the resolved IP, not the hostname."""
        from src.security.pinned_transport import _PinnedDnsTransport

        mock_dns.return_value = _mock_getaddrinfo(["93.184.216.34"])(
            "example.com", None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )

        pinned_ips_seen: list[str] = []

        # We'll intercept by checking the request that reaches the inner transport
        original_handle = httpx.AsyncHTTPTransport.handle_async_request

        async def mock_handle(self, request):
            # Verify the request URL uses the pinned IP, not the hostname
            pinned_ips_seen.append(request.url.host)
            # Return a minimal valid response
            return httpx.Response(200, request=request)

        with patch.object(httpx.AsyncHTTPTransport, "handle_async_request", mock_handle):
            transport = _PinnedDnsTransport()
            request = httpx.Request("GET", "http://example.com/test")
            await transport.handle_async_request(request)

        assert pinned_ips_seen == ["93.184.216.34"]
        await transport.aclose()

    @pytest.mark.asyncio
    @patch("src.security.url_safety.socket.getaddrinfo")
    async def test_blocks_private_ip(self, mock_dns):
        """Requests to hostnames that resolve to private IPs should raise ValueError."""
        from src.security.pinned_transport import _PinnedDnsTransport

        mock_dns.return_value = _mock_getaddrinfo(["10.0.0.1"])("evil.com", None, socket.AF_UNSPEC, socket.SOCK_STREAM)

        transport = _PinnedDnsTransport()
        request = httpx.Request("GET", "http://evil.com/steal")
        with pytest.raises(ValueError, match="private"):
            await transport.handle_async_request(request)
        await transport.aclose()

    @pytest.mark.asyncio
    async def test_literal_ip_validated_directly(self):
        """A literal IP in the URL should be validated without DNS lookup."""
        from src.security.pinned_transport import _PinnedDnsTransport

        transport = _PinnedDnsTransport()
        request = httpx.Request("GET", "http://93.184.216.34/test")

        async def mock_handle(self, req):
            return httpx.Response(200, request=req)

        with patch.object(httpx.AsyncHTTPTransport, "handle_async_request", mock_handle):
            resp = await transport.handle_async_request(request)
            assert resp.status_code == 200

        await transport.aclose()

    @pytest.mark.asyncio
    async def test_literal_private_ip_blocked(self):
        """A literal private IP should be blocked."""
        from src.security.pinned_transport import _PinnedDnsTransport

        transport = _PinnedDnsTransport()
        request = httpx.Request("GET", "http://127.0.0.1/healthz")
        with pytest.raises(ValueError, match="private"):
            await transport.handle_async_request(request)
        await transport.aclose()

    @pytest.mark.asyncio
    async def test_metadata_hostname_blocked_by_transport(self):
        """metadata.google.internal must be blocked even if DNS resolves to a public IP."""
        from src.security.pinned_transport import _PinnedDnsTransport

        transport = _PinnedDnsTransport()
        request = httpx.Request("GET", "http://metadata.google.internal/computeMetadata/v1/")
        with pytest.raises(ValueError, match="blocked hostname"):
            await transport.handle_async_request(request)
        await transport.aclose()

    @pytest.mark.asyncio
    @patch("src.security.url_safety.socket.getaddrinfo")
    async def test_https_sets_sni_hostname(self, mock_dns):
        """For HTTPS URLs, the transport must set sni_hostname to the original hostname."""
        from src.security.pinned_transport import _PinnedDnsTransport

        mock_dns.return_value = _mock_getaddrinfo(["93.184.216.34"])(
            "example.com", None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )

        sni_seen: list[str] = []

        async def mock_handle(self, request):
            sni_seen.append(request.extensions.get("sni_hostname", ""))
            return httpx.Response(200, request=request)

        with patch.object(httpx.AsyncHTTPTransport, "handle_async_request", mock_handle):
            transport = _PinnedDnsTransport()
            request = httpx.Request("GET", "https://example.com/test")
            await transport.handle_async_request(request)
            await transport.aclose()

        assert sni_seen == ["example.com"]

    @pytest.mark.asyncio
    @patch("src.security.url_safety.socket.getaddrinfo")
    async def test_host_header_preserved(self, mock_dns):
        """The Host header should contain the original hostname, not the pinned IP."""
        from src.security.pinned_transport import _PinnedDnsTransport

        mock_dns.return_value = _mock_getaddrinfo(["93.184.216.34"])(
            "example.com", None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )

        host_seen: list[str] = []

        async def mock_handle(self, request):
            host_seen.append(request.headers.get("host", ""))
            return httpx.Response(200, request=request)

        with patch.object(httpx.AsyncHTTPTransport, "handle_async_request", mock_handle):
            transport = _PinnedDnsTransport()
            request = httpx.Request("GET", "http://example.com/test")
            await transport.handle_async_request(request)
            await transport.aclose()

        assert host_seen == ["example.com"]


# ── safe_fetch ─────────────────────────────────────────────────────


class TestSafeFetch:
    def setup_method(self):
        reset_cache()

    @pytest.mark.asyncio
    @patch("src.security.url_safety.socket.getaddrinfo")
    async def test_basic_fetch(self, mock_dns):
        """safe_fetch should return a 200 response for a safe URL."""
        mock_dns.return_value = _mock_getaddrinfo(["93.184.216.34"])(
            "example.com", None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )

        async def mock_handle(self, request):
            return httpx.Response(200, text="OK", request=request)

        with patch.object(httpx.AsyncHTTPTransport, "handle_async_request", mock_handle):
            resp = await safe_fetch("http://example.com/test")

        assert resp.status_code == 200
        assert resp.text == "OK"

    @pytest.mark.asyncio
    async def test_blocks_private_ip(self):
        """safe_fetch should raise ValueError for private IPs."""
        with pytest.raises(ValueError, match="private"):
            await safe_fetch("http://127.0.0.1/healthz")

    @pytest.mark.asyncio
    async def test_blocks_invalid_scheme(self):
        """safe_fetch should reject non-HTTP schemes."""
        with pytest.raises(ValueError, match="Unsupported"):
            await safe_fetch("ftp://example.com/file")

    @pytest.mark.asyncio
    @patch("src.security.url_safety.socket.getaddrinfo")
    async def test_redirect_to_private_blocked(self, mock_dns):
        """A redirect from a safe URL to a private IP should be blocked."""
        # First call resolves safe domain, second resolves unsafe redirect
        call_count = 0
        original_resolve = _mock_getaddrinfo(["93.184.216.34"])

        def side_effect(host, port, family, type_, proto=0, flags=0):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return original_resolve("safe.com", port, family, type_)
            # Redirect target resolves to private IP
            return _mock_getaddrinfo(["10.0.0.1"])("evil.com", port, family, type_)

        mock_dns.side_effect = side_effect

        async def mock_handle(self, request):
            if "safe.com" in str(request.url) or request.headers.get("host") == "safe.com":
                # Redirect to private IP
                return httpx.Response(
                    302,
                    headers={"Location": "http://evil.com/steal"},
                    request=request,
                )
            return httpx.Response(200, request=request)

        with patch.object(httpx.AsyncHTTPTransport, "handle_async_request", mock_handle):
            with pytest.raises(ValueError, match="private"):
                await safe_fetch("http://safe.com/page", follow_redirects=True)

    @pytest.mark.asyncio
    @patch("src.security.url_safety.socket.getaddrinfo")
    async def test_redirect_chain_safe(self, mock_dns):
        """A redirect chain where all hops are safe should succeed."""
        mock_dns.return_value = _mock_getaddrinfo(["93.184.216.34"])(
            "example.com", None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )

        call_count = 0

        async def mock_handle(self, request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(
                    301,
                    headers={"Location": "http://example.com/final"},
                    request=request,
                )
            return httpx.Response(200, text="final page", request=request)

        with patch.object(httpx.AsyncHTTPTransport, "handle_async_request", mock_handle):
            resp = await safe_fetch("http://example.com/old", follow_redirects=True)

        assert resp.status_code == 200
        assert resp.text == "final page"

    @pytest.mark.asyncio
    async def test_url_length_limit(self):
        """URLs exceeding 2000 chars should be rejected."""
        long_url = "http://example.com/" + "a" * 2100
        with pytest.raises(ValueError, match="exceeds"):
            await safe_fetch(long_url)

    @pytest.mark.asyncio
    @patch("src.security.url_safety.socket.getaddrinfo")
    async def test_post_method(self, mock_dns):
        """safe_fetch should support POST method."""
        mock_dns.return_value = _mock_getaddrinfo(["93.184.216.34"])(
            "example.com", None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )

        bodies_seen: list[bytes] = []

        async def mock_handle(self, request):
            # Read the body
            body = b""
            async for chunk in request.stream:
                body += chunk if isinstance(chunk, bytes) else chunk.encode()
            bodies_seen.append(body)
            return httpx.Response(200, request=request)

        with patch.object(httpx.AsyncHTTPTransport, "handle_async_request", mock_handle):
            resp = await safe_fetch(
                "http://example.com/api",
                method="POST",
                content=b"hello",
                headers={"Content-Type": "text/plain"},
            )

        assert resp.status_code == 200
        assert bodies_seen == [b"hello"]
