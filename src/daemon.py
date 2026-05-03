"""Cross-platform daemon/service manager for MyClaw.

Supports:
- Linux: systemd
- macOS: launchd
- Windows: schtasks
"""

from __future__ import annotations

import getpass
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path


class DaemonManager:
    """Cross-platform daemon/service manager for MyClaw."""

    def __init__(self):
        self._platform = self.get_platform()
        self._python_path = sys.executable
        self._project_dir = Path(__file__).parent.parent.resolve()
        self._current_user = getpass.getuser()

    def get_platform(self) -> str:
        """Detect the current platform."""
        system = platform.system().lower()
        if system == "linux":
            return "systemd"
        elif system == "darwin":
            return "launchd"
        elif system == "windows":
            return "schtasks"
        else:
            raise NotImplementedError(f"Platform {system} is not supported")

    def install(self) -> None:
        """Install the MyClaw service."""
        if self._platform == "systemd":
            self._install_systemd()
        elif self._platform == "launchd":
            self._install_launchd()
        elif self._platform == "schtasks":
            self._install_schtasks()

    def uninstall(self) -> None:
        """Uninstall the MyClaw service."""
        if self._platform == "systemd":
            self._uninstall_systemd()
        elif self._platform == "launchd":
            self._uninstall_launchd()
        elif self._platform == "schtasks":
            self._uninstall_schtasks()

    def status(self) -> None:
        """Check the status of the MyClaw service."""
        if self._platform == "systemd":
            self._status_systemd()
        elif self._platform == "launchd":
            self._status_launchd()
        elif self._platform == "schtasks":
            self._status_schtasks()

    # ========== systemd (Linux) ==========

    def _get_systemd_service_path(self) -> Path:
        return Path("/etc/systemd/system/myclaw.service")

    def _generate_systemd_service(self) -> str:
        env_entries = os.environ.get("MYCLAW_ENV", "")
        return f"""[Unit]
Description=MyClaw AI Assistant
After=network.target

[Service]
Type=simple
User={self._current_user}
WorkingDirectory={self._project_dir}
ExecStart={shlex.quote(self._python_path)} -m src.main
Restart=on-failure
RestartSec=5
{f"Environment={env_entries}" if env_entries else ""}

[Install]
WantedBy=multi-user.target
"""

    def _install_systemd(self) -> None:
        service_path = self._get_systemd_service_path()
        service_content = self._generate_systemd_service()

        print("Installing MyClaw systemd service...")
        print(f"Service file: {service_path}")
        print()
        print("Run the following commands to complete the installation:")
        print(f"  echo '{service_content}' | sudo tee {service_path}")
        print("  sudo systemctl daemon-reload")
        print("  sudo systemctl enable myclaw")
        print("  sudo systemctl start myclaw")

    def _uninstall_systemd(self) -> None:
        print("Uninstalling MyClaw systemd service...")
        print("Run the following commands:")
        print("  sudo systemctl stop myclaw")
        print("  sudo systemctl disable myclaw")
        print(f"  sudo rm {self._get_systemd_service_path()}")

    def _status_systemd(self) -> None:
        result = subprocess.run(
            ["systemctl", "is-active", "myclaw"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"Service status: {result.stdout.strip()}")
        else:
            print("Service is not active or not installed.")

    # ========== launchd (macOS) ==========

    def _get_launchd_plist_path(self) -> Path:
        return Path.home() / "Library" / "LaunchAgents" / "com.myclaw.agent.plist"

    def _generate_launchd_plist(self) -> str:
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.myclaw.agent</string>
    <key>ProgramArguments</key>
    <array><string>{self._python_path}</string><string>-m</string><string>src.main</string></array>
    <key>WorkingDirectory</key><string>{self._project_dir}</string>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>{self._project_dir}/data/myclaw.log</string>
    <key>StandardErrorPath</key><string>{self._project_dir}/data/myclaw.log</string>
</dict>
</plist>
"""

    def _install_launchd(self) -> None:
        plist_path = self._get_launchd_plist_path()
        plist_content = self._generate_launchd_plist()

        print("Installing MyClaw launchd agent...")
        print(f"Plist file: {plist_path}")
        print()
        print("Run the following commands to complete the installation:")
        print(f"  mkdir -p {plist_path.parent}")
        print(f"  echo '{plist_content}' > {plist_path}")
        print(f"  launchctl load {plist_path}")

    def _uninstall_launchd(self) -> None:
        plist_path = self._get_launchd_plist_path()
        print("Uninstalling MyClaw launchd agent...")
        print("Run the following commands:")
        print(f"  launchctl unload {plist_path}")
        print(f"  rm {plist_path}")

    def _status_launchd(self) -> None:
        result = subprocess.run(
            ["launchctl", "list", "com.myclaw.agent"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("Agent is loaded and running.")
        else:
            print("Agent is not loaded or not installed.")

    # ========== schtasks (Windows) ==========

    def _install_schtasks(self) -> None:
        cmd = [
            "schtasks",
            "/create",
            "/tn",
            "MyClaw",
            "/tr",
            f'"{self._python_path}" -m src.main',
            "/sc",
            "onstart",
            "/ru",
            self._current_user,
            "/rl",
            "highest",
            "/f",
        ]

        print("Installing MyClaw scheduled task...")
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
        cmd = ["schtasks", "/delete", "/tn", "MyClaw", "/f"]

        print("Uninstalling MyClaw scheduled task...")
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
            ["schtasks", "/query", "/tn", "MyClaw"],
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
    manager = DaemonManager()
    action = sys.argv[2] if len(sys.argv) > 2 else "status"

    if action == "install":
        manager.install()
    elif action == "uninstall":
        manager.uninstall()
    elif action == "status":
        manager.status()
    else:
        print(f"Usage: myclaw-daemon [install|uninstall|status]")
        sys.exit(1)


if __name__ == "__main__":
    main_daemon()
