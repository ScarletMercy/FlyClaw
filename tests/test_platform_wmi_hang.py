"""回归测试：Windows 上不得走 platform.system/release/win32_ver（WMI 挂死）。

Py3.13 + Win 上 platform.system() / release() / win32_ver() 经 WMI 查询，服务抽风时
无限挂起。prompt._build_environment_hints 的 Windows 分支与 daemon.get_platform 必须
改走 sys.getwindowsversion() / sys.platform。本测试把 platform.* 调用变成 fail，
锁定"Windows 路径不再触碰 platform.*"这一行为不变量。
"""

from __future__ import annotations

import platform as platform_mod
import sys
from types import SimpleNamespace

import pytest

from src.daemon import DaemonManager
from src.prompt import _build_environment_hints


def _boom(*_a, **_k):
    pytest.fail("platform.system/release/win32_ver 被调用 — WMI 挂死路径")


def test_env_hints_win32_avoids_platform_wmi(monkeypatch):
    """Windows 分支建环境提示不得调 platform.*。"""
    monkeypatch.setattr(sys, "platform", "win32")
    # sys.getwindowsversion 仅 Windows 存在；非 Win 平台用 raising=False 注入假实现
    monkeypatch.setattr(sys, "getwindowsversion", lambda: SimpleNamespace(build=26120), raising=False)
    monkeypatch.setattr(platform_mod, "release", _boom)
    monkeypatch.setattr(platform_mod, "system", _boom)
    monkeypatch.setattr(platform_mod, "win32_ver", _boom)

    hints = _build_environment_hints()
    assert any("Windows" in h and "build" in h for h in hints)


def test_get_platform_win32_avoids_platform_system(monkeypatch):
    """get_platform 在 Win 上返回 schtasks，且不调 platform.system。"""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(platform_mod, "system", _boom)

    # 绕过 __init__（会调 home_dir/getpass，与平台检测无关）
    mgr = DaemonManager.__new__(DaemonManager)
    assert mgr.get_platform() == "schtasks"
