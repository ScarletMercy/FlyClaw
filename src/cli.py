"""MyClaw CLI 管理命令。

用法:
  myclaw                启动服务器（默认）
  myclaw doctor         运行系统诊断
  myclaw status         显示系统状态
  myclaw sessions       列出活跃会话
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_config():
    from src.config import load_config

    return load_config()


def cmd_doctor(args):
    print("MyClaw 系统诊断\n" + "=" * 40)
    errors = []
    warnings = []

    try:
        config = _load_config()
        print("[通过] 配置已加载")
    except Exception as e:
        errors.append(f"配置: {e}")
        print(f"[失败] 配置: {e}")
        return 1

    if config.model.api_key:
        masked = config.model.api_key[:8] + "..." if len(config.model.api_key) > 8 else "***"
        print(f"[通过] 模型 API 密钥已设置 ({masked})")
    else:
        warnings.append("模型 API 密钥未设置 — 将使用环境变量或默认值")
        print("[警告] 模型 API 密钥未设置")

    if config.model.provider and config.model.name:
        print(f"[通过] 模型: {config.model.provider}/{config.model.name}")

    print(f"[通过] 网关: {config.gateway.host}:{config.gateway.port}")
    if not config.gateway.auth_token:
        warnings.append("网关认证令牌为空")
        print("[警告] 网关认证令牌为空 — 认证已禁用")

    db_path = Path(config.checkpointer.path)
    if db_path.parent.exists():
        print(f"[通过] 会话存储目录: {db_path.parent}")
    else:
        warnings.append(f"会话存储目录不存在: {db_path.parent}")
        print(f"[警告] 会话存储目录不存在: {db_path.parent}")

    if config.cron.enabled:
        cron_path = Path(config.cron.store_path)
        if cron_path.parent.exists():
            print(f"[通过] 定时任务目录: {cron_path.parent}")
        else:
            print(f"[警告] 定时任务目录不存在: {cron_path.parent}")

    if config.memory.enabled:
        print(f"[通过] 记忆系统已启用 (db={config.memory.db_path})")

    print(f"[通过] 命令执行: 已启用={config.tools.exec.enabled}, 审批模式={config.tools.exec.approval_mode}")
    print(f"[通过] 网页搜索: 已启用={config.tools.web_search.enabled}")

    if config.security.enabled:
        print("[通过] 安全检查已启用")

    print(f"\n{'=' * 40}")
    if errors:
        print(f"结果: {len(errors)} 个错误, {len(warnings)} 个警告")
        for e in errors:
            print(f"  错误: {e}")
        return 1
    elif warnings:
        print(f"结果: 通过 ({len(warnings)} 个警告)")
        for w in warnings:
            print(f"  警告: {w}")
        return 0
    else:
        print("结果: 所有检查通过")
        return 0


def cmd_status(args):
    config = _load_config()
    print("MyClaw 系统状态\n" + "=" * 40)
    print(f"模型:       {config.model.provider}/{config.model.name}")
    print(f"网关:       {config.gateway.host}:{config.gateway.port}")
    print(f"QQ:         {'已启用' if config.channels.qq.enabled else '未启用'}")
    print(f"定时任务:   {'已启用' if config.cron.enabled else '未启用'}")
    print(f"记忆系统:   {'已启用' if config.memory.enabled else '未启用'}")
    print(f"技能:       {'已启用' if config.skills.enabled else '未启用'}")
    print(f"插件:       {'已启用' if config.plugins.enabled else '未启用'}")
    print(f"命令执行:   审批模式={config.tools.exec.approval_mode}")
    print(f"会话:       作用域={config.session.scope}")

    import urllib.request

    try:
        url = f"http://{config.gateway.host}:{config.gateway.port}/healthz"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {config.gateway.auth_token}"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            print(f"\n服务状态:   运行中 (healthz={resp.status})")
    except Exception:
        print(f"\n服务状态:   未运行 (无法连接 {config.gateway.host}:{config.gateway.port})")


def cmd_sessions(args):
    config = _load_config()
    db_path = Path(config.checkpointer.path)
    if not db_path.exists():
        print("未找到会话（数据库不存在）")
        return 0

    from src.agent.state import StateStore

    store = StateStore(str(db_path))
    try:
        threads = store.list_threads()
        if not threads:
            print("未找到会话")
            return 0
        print(f"会话列表 ({len(threads)}):\n")
        for tid in threads:
            state = store.load(tid)
            msg_count = len(state.messages) if state else 0
            print(f"  {tid}: {msg_count} 条消息")
    finally:
        store.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="myclaw", description="MyClaw AI 助手")
    parser.add_argument("--version", action="version", version="myclaw 0.1.0")
    sub = parser.add_subparsers(dest="command", help="管理命令")

    sub.add_parser("doctor", help="运行系统诊断")
    sub.add_parser("status", help="显示系统状态")
    sub.add_parser("sessions", help="列出活跃会话")
    sub.add_parser("setup", help="交互式配置向导")

    return parser


def cli_main():
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
        from src.main import main

        main()
