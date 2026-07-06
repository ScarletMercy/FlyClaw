"""Cross-platform daemon/service manager for flyclaw.

Supports:
- Linux: systemd
- macOS: launchd
- Windows: schtasks
"""

from __future__ import annotations

import argparse
import getpass
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


class DaemonManager:
    """Cross-platform daemon/service manager for flyclaw."""

    def __init__(self, instance: int | None = None):
        self._instance = instance
        self._platform = self.get_platform()
        self._python_path = sys.executable
        from src.instance import home_dir

        self._project_dir = home_dir(instance)
        self._current_user = getpass.getuser()

    @property
    def _service_name(self) -> str:
        """Service name including instance suffix."""
        if self._instance is not None:
            return f"flyclaw-{self._instance}"
        return "flyclaw"

    @property
    def _exec_args(self) -> list[str]:
        """Raw command-line arguments (no shell quoting)."""
        args = [self._python_path, "-m", "src.main"]
        if self._instance is not None:
            args.append(str(self._instance))
        return args

    def get_platform(self) -> str:
        """Detect the current platform.

        用 sys.platform 不走 platform.system()——后者在 Py3.13/Win 经 WMI，
        服务抽风时会挂死。注意 sys.platform 值与 platform.system().lower() 不同：
        Win 上是 'win32' 而非 'windows'。
        """
        if sys.platform == "linux":
            return "systemd"
        elif sys.platform == "darwin":
            return "launchd"
        elif sys.platform == "win32":
            return "schtasks"
        else:
            raise NotImplementedError(f"Platform {sys.platform} is not supported")

    def install(self) -> None:
        """Install the flyclaw service."""
        if self._platform == "systemd":
            self._install_systemd()
        elif self._platform == "launchd":
            self._install_launchd()
        elif self._platform == "schtasks":
            self._install_schtasks()

    def uninstall(self) -> None:
        """Uninstall the flyclaw service."""
        if self._platform == "systemd":
            self._uninstall_systemd()
        elif self._platform == "launchd":
            self._uninstall_launchd()
        elif self._platform == "schtasks":
            self._uninstall_schtasks()

    def status(self) -> None:
        """Check the status of the flyclaw service."""
        if self._platform == "systemd":
            self._status_systemd()
        elif self._platform == "launchd":
            self._status_launchd()
        elif self._platform == "schtasks":
            self._status_schtasks()

    # ========== systemd (Linux) ==========

    def _get_systemd_service_path(self) -> Path:
        return Path(f"/etc/systemd/system/{self._service_name}.service")

    def _generate_systemd_service(self) -> str:
        env_entries = os.environ.get("FLYCLAW_ENV", "")
        exec_start = " ".join(shlex.quote(a) for a in self._exec_args)
        desc = "flyclaw AI Assistant" + (f" (instance {self._instance})" if self._instance else "")
        return f"""[Unit]
Description={desc}
After=network.target

[Service]
Type=simple
User={self._current_user}
WorkingDirectory={self._project_dir}
ExecStart={exec_start}
Restart=on-failure
RestartSec=5
{f"Environment={env_entries}" if env_entries else ""}

[Install]
WantedBy=multi-user.target
"""

    def _install_systemd(self) -> None:
        service_path = self._get_systemd_service_path()
        service_content = self._generate_systemd_service()

        tmp = Path(tempfile.gettempdir()) / f"{self._service_name}.service"
        tmp.write_text(service_content, encoding="utf-8")

        print(f"Installing {self._service_name} systemd service...")
        print(f"Service file: {service_path}")
        print()
        print("Run the following commands to complete the installation:")
        print(f"  sudo cp {tmp} {service_path}")
        print("  sudo systemctl daemon-reload")
        print(f"  sudo systemctl enable {self._service_name}")
        print(f"  sudo systemctl start {self._service_name}")

    def _uninstall_systemd(self) -> None:
        print(f"Uninstalling {self._service_name} systemd service...")
        print("Run the following commands:")
        print(f"  sudo systemctl stop {self._service_name}")
        print(f"  sudo systemctl disable {self._service_name}")
        print(f"  sudo rm {self._get_systemd_service_path()}")

        tmp = Path(tempfile.gettempdir()) / f"{self._service_name}.service"
        if tmp.exists():
            tmp.unlink()

    def _status_systemd(self) -> None:
        result = subprocess.run(
            ["systemctl", "is-active", self._service_name],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"Service status: {result.stdout.strip()}")
        else:
            print("Service is not active or not installed.")

    # ========== launchd (macOS) ==========

    def _get_launchd_label(self) -> str:
        if self._instance is not None:
            return f"com.flyclaw.agent.{self._instance}"
        return "com.flyclaw.agent"

    def _get_launchd_plist_path(self) -> Path:
        return Path.home() / "Library" / "LaunchAgents" / f"{self._get_launchd_label()}.plist"

    def _generate_launchd_plist(self) -> str:
        from src.instance import data_dir

        log_dir = data_dir(self._instance)
        log_dir.mkdir(parents=True, exist_ok=True)
        label = self._get_launchd_label()
        args = "".join(f"<string>{a}</string>" for a in self._exec_args)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>{label}</string>
    <key>ProgramArguments</key>
    <array>{args}</array>
    <key>WorkingDirectory</key><string>{self._project_dir}</string>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>{log_dir / "flyclaw.log"}</string>
    <key>StandardErrorPath</key><string>{log_dir / "flyclaw.log"}</string>
</dict>
</plist>
"""

    def _install_launchd(self) -> None:
        plist_path = self._get_launchd_plist_path()
        plist_content = self._generate_launchd_plist()

        tmp = Path(tempfile.gettempdir()) / f"{self._get_launchd_label()}.plist"
        tmp.write_text(plist_content, encoding="utf-8")

        print(f"Installing {self._service_name} launchd agent...")
        print(f"Plist file: {plist_path}")
        print()
        print("Run the following commands to complete the installation:")
        print(f"  mkdir -p {plist_path.parent}")
        print(f"  cp {tmp} {plist_path}")
        print(f"  launchctl load {plist_path}")

    def _uninstall_launchd(self) -> None:
        plist_path = self._get_launchd_plist_path()
        print(f"Uninstalling {self._service_name} launchd agent...")
        print("Run the following commands:")
        print(f"  launchctl unload {plist_path}")
        print(f"  rm {plist_path}")

        tmp = Path(tempfile.gettempdir()) / f"{self._get_launchd_label()}.plist"
        if tmp.exists():
            tmp.unlink()

    def _status_launchd(self) -> None:
        result = subprocess.run(
            ["launchctl", "list", self._get_launchd_label()],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("Agent is loaded and running.")
        else:
            print("Agent is not loaded or not installed.")

    # ========== schtasks (Windows) ==========

    def _install_schtasks(self) -> None:
        exec_cmd = f'"{self._python_path}" -m src.main'
        if self._instance is not None:
            exec_cmd += f" {self._instance}"
        cmd = [
            "schtasks",
            "/create",
            "/tn",
            self._service_name,
            "/tr",
            exec_cmd,
            "/sc",
            "onstart",
            "/ru",
            self._current_user,
            "/rl",
            "highest",
            "/f",
        ]

        print(f"Installing {self._service_name} scheduled task...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("Scheduled task created successfully.")
        else:
            print(f"Failed to create scheduled task: {result.stderr}")

    def _uninstall_schtasks(self) -> None:
        cmd = ["schtasks", "/delete", "/tn", self._service_name, "/f"]

        print(f"Uninstalling {self._service_name} scheduled task...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("Scheduled task deleted successfully.")
        else:
            print(f"Failed to delete scheduled task: {result.stderr}")

    def _status_schtasks(self) -> None:
        result = subprocess.run(
            ["schtasks", "/query", "/tn", self._service_name],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("Scheduled task exists:")
            print(result.stdout)
        else:
            print("Scheduled task does not exist.")


def main_daemon():
    """CLI entry point for daemon management."""
    from src.instance import parse_instance_from_argv, set_instance

    n = parse_instance_from_argv()
    set_instance(n)

    parser = argparse.ArgumentParser(
        prog="flyclaw-daemon",
        description="flyclaw 守护进程管理工具",
    )
    parser.add_argument(
        "action",
        choices=["install", "uninstall", "status"],
        nargs="?",
        default="status",
        help="要执行的操作 (默认: status)",
    )
    args = parser.parse_args()
    manager = DaemonManager(instance=n)

    if args.action == "install":
        manager.install()
    elif args.action == "uninstall":
        manager.uninstall()
    else:
        manager.status()


if __name__ == "__main__":
    main_daemon()
