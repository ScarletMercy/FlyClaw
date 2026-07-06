"""Channel package — public names are lazy-loaded.

Importing this package no longer eagerly pulls in weixin (→ aiohttp), which on
Python 3.13 + Windows can hang at ``platform.system()`` → WMI.  Channel classes
are resolved on first attribute access via ``__getattr__``; direct submodule
imports (``from src.channels.qq import QQChannel``) are unaffected and still
preferred.

Background: ``aiohttp/helpers.py`` calls ``platform.system()`` at import time;
on Python 3.13 that routes through ``platform.win32_ver()`` → ``_wmi_query()``,
which can block indefinitely when the WMI service is unresponsive.  The setup
wizard's scan-to-configure (``src.channels.qq_onboard``) only needs httpx +
cryptography, so decoupling it from the weixin/aiohttp import keeps onboarding
working even when WMI is hung.
"""

__all__ = [
    "Channel",
    "QQChannel",
    "WeixinChannel",
]


def __getattr__(name: str):
    if name == "Channel":
        from .base import Channel

        return Channel
    if name == "QQChannel":
        from .qq import QQChannel

        return QQChannel
    if name == "WeixinChannel":
        from .weixin import WeixinChannel

        return WeixinChannel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # Keep dir(src.channels) listing the public channel names alongside the
    # standard module attributes, so introspection / IDE completion is preserved.
    return sorted(set(globals()) | set(__all__))
