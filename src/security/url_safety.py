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

        # Always block known metadata hostnames
        if hostname in _BLOCKED_HOSTNAMES:
            return False, f"blocked hostname: {hostname}"

        # Resolve hostname to IP(s)
        try:
            addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except socket.gaierror:
            return False, f"DNS resolution failed: {hostname}"

        for _family, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                continue

            # Always block cloud metadata IPs
            if _is_always_blocked(ip):
                return False, f"cloud metadata address: {hostname} -> {ip_str}"

            # Check private IPs (unless allowed by config)
            if not _allow_private_urls() and _is_private_ip(ip):
                return False, f"private/internal address: {hostname} -> {ip_str}"

        return True, ""

    except Exception as exc:
        return False, f"URL safety check error: {exc}"
