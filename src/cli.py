"""CLI management commands for MyClaw.

Usage:
  myclaw                Start the server (default)
  myclaw doctor         Run diagnostics
  myclaw status         Show system status
  myclaw sessions       List active sessions
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def _load_config():
    from src.config import load_config

    return load_config()


def cmd_doctor(args):
    """Run system diagnostics."""
    print("MyClaw Diagnostics\n" + "=" * 40)
    errors = []
    warnings = []

    # Check config
    try:
        config = _load_config()
        print("[OK] Config loaded")
    except Exception as e:
        errors.append(f"Config: {e}")
        print(f"[FAIL] Config: {e}")
        return 1

    # Check model config
    if config.model.api_key:
        masked = config.model.api_key[:8] + "..." if len(config.model.api_key) > 8 else "***"
        print(f"[OK] Model API key set ({masked})")
    else:
        warnings.append("Model API key not set — will use env variable or default")
        print("[WARN] Model API key not set")

    if config.model.provider and config.model.name:
        print(f"[OK] Model: {config.model.provider}/{config.model.name}")

    # Check gateway
    print(f"[OK] Gateway: {config.gateway.host}:{config.gateway.port}")
    if not config.gateway.auth_token:
        warnings.append("Gateway auth_token is empty")
        print("[WARN] Gateway auth_token is empty — all auth disabled")

    # Check Feishu
    if config.channels.feishu.enabled:
        if config.channels.feishu.app_id and config.channels.feishu.app_secret:
            print(f"[OK] Feishu configured (domain={config.channels.feishu.domain})")
        else:
            errors.append("Feishu enabled but app_id or app_secret is empty")
            print("[FAIL] Feishu enabled but credentials missing")
    else:
        print("[--] Feishu disabled")

    # Check database files
    db_path = Path(config.checkpointer.path)
    if db_path.parent.exists():
        print(f"[OK] Checkpoint dir: {db_path.parent}")
    else:
        warnings.append(f"Checkpoint dir missing: {db_path.parent}")
        print(f"[WARN] Checkpoint dir missing: {db_path.parent}")

    if config.cron.enabled:
        cron_path = Path(config.cron.store_path)
        if cron_path.parent.exists():
            print(f"[OK] Cron dir: {cron_path.parent}")
        else:
            print(f"[WARN] Cron dir missing: {cron_path.parent}")

    # Check memory
    if config.memory.enabled:
        print(f"[OK] Memory enabled (db={config.memory.db_path})")

    # Check TTS
    if config.tts.enabled:
        print(f"[OK] TTS enabled (provider={config.tts.provider})")

    # Check tools
    print(f"[OK] Exec: enabled={config.tools.exec.enabled}, approval={config.tools.exec.approval_mode}")
    print(f"[OK] Web search: enabled={config.tools.web_search.enabled}")

    # Security
    if config.security.enabled:
        print("[OK] Security checks enabled")

    # Summary
    print(f"\n{'=' * 40}")
    if errors:
        print(f"Result: {len(errors)} error(s), {len(warnings)} warning(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    elif warnings:
        print(f"Result: OK ({len(warnings)} warning(s))")
        for w in warnings:
            print(f"  WARN: {w}")
        return 0
    else:
        print("Result: All checks passed")
        return 0


def cmd_status(args):
    """Show running system status."""
    config = _load_config()
    print("MyClaw System Status\n" + "=" * 40)
    print(f"Model:      {config.model.provider}/{config.model.name}")
    print(f"Gateway:    {config.gateway.host}:{config.gateway.port}")
    print(f"Feishu:     {'enabled' if config.channels.feishu.enabled else 'disabled'}")
    print(f"Cron:       {'enabled' if config.cron.enabled else 'disabled'}")
    print(f"Memory:     {'enabled' if config.memory.enabled else 'disabled'}")
    print(f"TTS:        {'enabled' if config.tts.enabled else 'disabled'}")
    print(f"Skills:     {'enabled' if config.skills.enabled else 'disabled'}")
    print(f"Plugins:    {'enabled' if config.plugins.enabled else 'disabled'}")
    print(f"Exec:       approval={config.tools.exec.approval_mode}")
    print(f"Session:    scope={config.session.scope}")

    # Check if server is running by probing health endpoint
    import urllib.request

    try:
        url = f"http://{config.gateway.host}:{config.gateway.port}/healthz"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {config.gateway.auth_token}"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            print(f"\nServer:     RUNNING (healthz={resp.status})")
    except Exception:
        print(f"\nServer:     NOT RUNNING (could not reach {config.gateway.host}:{config.gateway.port})")


def cmd_sessions(args):
    """List active sessions from checkpoint database."""
    config = _load_config()
    db_path = Path(config.checkpointer.path)
    if not db_path.exists():
        print("No sessions found (checkpoint database does not exist)")
        return 0

    async def _list_sessions():
        import aiosqlite

        async with aiosqlite.connect(str(db_path)) as conn:
            # LangGraph checkpoint tables: checkpoints, writes
            try:
                cursor = await conn.execute(
                    "SELECT thread_id, parent_checkpoint_id FROM checkpoints ORDER BY thread_id"
                )
                rows = await cursor.fetchall()
                if not rows:
                    print("No sessions found")
                    return
                # Count messages per thread
                sessions = {}
                for row in rows:
                    tid = row[0]
                    sessions[tid] = sessions.get(tid, 0) + 1
                print(f"Sessions ({len(sessions)}):\n")
                for tid, count in sorted(sessions.items()):
                    print(f"  {tid}: {count} checkpoints")
            except Exception as e:
                print(f"Error reading sessions: {e}")

    asyncio.run(_list_sessions())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="myclaw", description="MyClaw AI Assistant")
    parser.add_argument("--version", action="version", version="myclaw 0.1.0")
    sub = parser.add_subparsers(dest="command", help="Management commands")

    sub.add_parser("doctor", help="Run system diagnostics")
    sub.add_parser("status", help="Show system status")
    sub.add_parser("sessions", help="List active sessions")
    sub.add_parser("setup", help="Interactive configuration wizard")

    return parser


def cli_main():
    """CLI entry point — handles both management commands and server start."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "doctor":
        sys.exit(cmd_doctor(args))
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "sessions":
        sys.exit(cmd_sessions(args))
    elif args.command == "setup":
        from src.setup import run_wizard

        run_wizard()
    else:
        # Default: start the server
        from src.main import main

        main()
