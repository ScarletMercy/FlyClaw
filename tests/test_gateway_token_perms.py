"""Token file permission restriction: cross-platform.

On POSIX, chmod 0600 restricts to owner. On Windows, chmod is a no-op for the
0600 semantic, so we must use icacls to actually restrict access. This tests
the platform dispatch without depending on the real platform the test runs on
(monkeypatch sys.platform + the subprocess/shim call).
"""

from unittest.mock import patch

from src.gateway import _restrict_token_file


def test_posix_uses_chmod(tmp_path):
    f = tmp_path / "gateway_token"
    f.write_text("x")
    with patch("src.gateway.sys") as mock_sys, patch("src.gateway.os.chmod") as mock_chmod:
        mock_sys.platform = "linux"
        _restrict_token_file(f)
    mock_chmod.assert_called_once_with(f, 0o600)


def test_windows_uses_icacls(tmp_path):
    f = tmp_path / "gateway_token"
    f.write_text("x")
    with (
        patch("src.gateway.sys") as mock_sys,
        patch("src.gateway.getpass") as mock_getpass,
        patch("src.gateway.subprocess.run") as mock_run,
    ):
        mock_sys.platform = "win32"
        mock_getpass.getuser.return_value = "alice"
        _restrict_token_file(f)
    # 必须调 icacls 收紧权限(不是 chmod,chmod 在 Windows 无效)
    assert mock_run.called, "Windows 必须用 icacls,不能只靠 chmod"
    args = mock_run.call_args[0][0]
    assert args[0] == "icacls", f"应调 icacls,实际: {args[0]}"
    assert "/inheritance:r" in args, "必须移除继承的权限"
    assert any("alice" in a and ":R" in a for a in args), "必须只授予当前用户读权限"
