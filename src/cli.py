"""flyclaw CLI 管理命令。

用法:
  flyclaw                启动服务器（默认）
  flyclaw doctor         运行系统诊断
  flyclaw status         显示系统状态
  flyclaw sessions       列出活跃会话
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from src.version import __version__

from src.gateway import GATEWAY_HOST


def _load_config():
    from src.config import load_config
    from src.instance import config_path

    return load_config(config_path())


def cmd_doctor(args):
    print("flyclaw 系统诊断\n" + "=" * 40)
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
        errors.append("模型 API 密钥未设置 — 请运行 flyclaw setup 配置")
        print("[错误] 模型 API 密钥未设置")

    if config.model.provider and config.model.name:
        print(f"[通过] 模型: {config.model.name}")

    # 验证模型是否可用
    if config.model.api_key:
        print("验证模型可用性中...")
        try:
            from src.agent.client import ChatClient

            async def _test_model():
                client = ChatClient(
                    base_url=config.model.base_url or "",
                    api_key=config.model.api_key,
                    model=config.model.name,
                )
                resp = await client.chat_simple([{"role": "user", "content": "你好"}])
                return True, resp

            success, msg = asyncio.run(_test_model())
            if success:
                print(f"[通过] 模型验证成功")
            else:
                errors.append(f"模型验证失败: {msg}")
                print(f"[错误] 模型验证失败: {msg}")
        except Exception as e:
            errors.append(f"模型验证失败: {e}")
            print(f"[错误] 模型验证失败: {e}")

    print(f"[通过] 网关: {GATEWAY_HOST}:{config.gateway.port}")
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
    print("flyclaw 系统状态\n" + "=" * 40)
    print(f"模型:       {config.model.name}")
    print(f"网关:       {GATEWAY_HOST}:{config.gateway.port}")
    print(f"QQ:         {'已启用' if config.channels.qq.enabled else '未启用'}")
    print(f"定时任务:   {'已启用' if config.cron.enabled else '未启用'}")
    print(f"记忆系统:   {'已启用' if config.memory.enabled else '未启用'}")
    print(f"技能:       {'已启用' if config.skills.enabled else '未启用'}")
    print(f"插件:       {'已启用' if config.plugins.enabled else '未启用'}")
    print(f"命令执行:   审批模式={config.tools.exec.approval_mode}")
    print(f"会话:       DM塌缩(私聊不按openid分), 空闲重置={config.session.idle_reset_minutes}分钟")

    import urllib.request

    try:
        url = f"http://{GATEWAY_HOST}:{config.gateway.port}/healthz"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {config.gateway.auth_token}"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            print(f"\n服务状态:   运行中 (healthz={resp.status})")
    except Exception:
        print(f"\n服务状态:   未运行 (无法连接 {GATEWAY_HOST}:{config.gateway.port})")


async def _cmd_sessions_async(args):
    config = _load_config()
    db_path = Path(config.checkpointer.path)
    if not db_path.exists():
        print("未找到会话（数据库不存在）")
        return 0

    from src.agent.state import StateStore

    store = StateStore(str(db_path))
    try:
        threads = await store.list_threads()
        if not threads:
            print("未找到会话")
            return 0
        print(f"会话列表 ({len(threads)}):\n")
        for tid in threads:
            state = await store.load(tid)
            msg_count = len(state.messages) if state else 0
            print(f"  {tid}: {msg_count} 条消息")
    finally:
        await store.close()
    return 0


def cmd_sessions(args):
    return asyncio.run(_cmd_sessions_async(args))


def _prompt_fallback_model():
    """交互式收集一个回退模型(复用 setup 的 PRESETS + _ask* 辅助,UX 与向导一致),返回 ModelFallback。

    _configure_fallbacks 的"添加"分支逻辑(setup.py:227-248)的 CLI 友好版本:
    不嵌在向导里,单独可调(`flyclaw model add` 用)。
    """
    from src.config import ModelFallback
    from src.setup import PRESETS, _ask, _ask_choice, _ask_int, _ask_yn

    choice = _ask_choice("  模型提供商", list(PRESETS.keys()), default="custom")
    preset = PRESETS[choice]
    if preset:
        provider = preset["provider"]
        name = _ask("  模型名称", default=preset["name"])
        base_url = _ask("  接口地址", default=preset["base_url"] or "")
        api_key = _ask(f"  API 密钥 ({preset['env_key']})", default="") if preset["env_key"] else ""
    else:
        provider = "openai"
        name = _ask("  模型名称", default="")
        base_url = _ask("  接口地址", default="")
        api_key = _ask("  API 密钥", default="")
    context_window = _ask_int("  上下文窗口大小 (tokens)", default=200000)
    multimodal = _ask_yn("  该回退模型是否支持多模态？(切换到它时视觉跟随)", default=False)
    return ModelFallback(
        provider=provider,
        name=name,
        base_url=base_url or None,
        api_key=api_key or None,
        context_window=context_window,
        multimodal=multimodal,
    )


def cmd_model(args):
    config = _load_config()
    model_list = [
        {
            "provider": config.model.provider,
            "name": config.model.name,
            "base_url": config.model.base_url,
            "api_key": config.model.api_key,
            "context_window": config.model.context_window,
            "multimodal": config.model.multimodal,
        },
    ]
    for fb in config.model.fallbacks or []:
        model_list.append(
            {
                "provider": fb.provider,
                "name": fb.name,
                "base_url": fb.base_url or config.model.base_url,
                "api_key": fb.api_key or config.model.api_key,
                "context_window": fb.context_window,
                "multimodal": fb.multimodal,
            }
        )

    sub = getattr(args, "model_command", None)

    # flyclaw model list
    if sub == "list":
        print(f"模型列表 ({len(model_list)}):")
        for i, m in enumerate(model_list):
            mm = " (多模态)" if m.get("multimodal") else ""
            print(f"  [{i}] {m['name']}{mm}")
        return 0

    if sub is None:
        print(f"模型列表 ({len(model_list)}):")
        for i, m in enumerate(model_list):
            mm = " (多模态)" if m.get("multimodal") else ""
            print(f"  [{i}] {m['name']}{mm}")
        print("\n用法:")
        print("  flyclaw model list              — 列出所有模型")
        print("  flyclaw model switch <id>       — 切换模型")
        print("  flyclaw model test              — 测试当前模型")
        print("  flyclaw model add               — 添加新模型")
        print("  flyclaw model config            — 交互式配置当前模型")
        return 0

    # flyclaw model switch <id>
    if sub == "switch":
        idx = getattr(args, "id", None)
        if idx is None or idx < 0 or idx >= len(model_list):
            print(f"无效 ID。使用 flyclaw model list 查看可用模型 (0-{len(model_list) - 1})。")
            return 1
        if idx == 0:
            print(f"[{idx}] {config.model.name} 已是当前主模型。")
            return 0
        # 交换语义:选中 fallback 升主,旧主模型降入该 fallback 位(模型不丢、不重)。
        # 修复前只把 fallback 拷进主模型、不动 fallbacks → 切完主模型与某 fallback 完全重复。
        fb_idx = idx - 1
        m = model_list[idx]  # base_url/api_key 已与主模型合并(fallback 为 None 时回退主模型)
        from src.config import ModelFallback, save_config

        old_primary = ModelFallback(
            provider=config.model.provider,
            name=config.model.name,
            base_url=config.model.base_url,
            api_key=config.model.api_key,
            context_window=config.model.context_window,
            multimodal=config.model.multimodal,
        )
        config.model.provider = m["provider"]
        config.model.name = m["name"]
        config.model.base_url = m["base_url"]
        config.model.api_key = m["api_key"]
        config.model.context_window = m["context_window"]
        config.model.multimodal = m["multimodal"]
        config.model.fallbacks[fb_idx] = old_primary
        save_config(config)
        print(f"已切换到 [{idx}] {m['name']}(旧主模型已降为回退)")
        return 0

    # flyclaw model test
    if sub == "test":
        print("测试当前模型中...")
        try:
            from src.agent.client import ChatClient

            async def _test():
                client = ChatClient(
                    base_url=config.model.base_url or "",
                    api_key=config.model.api_key or "",
                    model=config.model.name,
                )
                resp = await client.chat_simple([{"role": "user", "content": "你好"}])
                return True, resp

            success, msg = asyncio.run(_test())
            if success:
                print(f"[通过] 模型验证成功")
                print(f"响应: {msg[:100]}...")
            else:
                print(f"[错误] 模型验证失败: {msg}")
                return 1
        except Exception as e:
            print(f"[错误] 模型验证失败: {e}")
            return 1
        return 0

    # flyclaw model add
    if sub == "add":
        print("\n  添加回退模型\n  ────────────")
        fb = _prompt_fallback_model()
        config.model.fallbacks.append(fb)
        from src.config import save_config

        save_config(config)
        print(f"\n  已添加回退模型 {fb.name}")
        print("  提示:想设为主模型用 `flyclaw model switch`。")
        return 0

    # flyclaw model config
    if sub == "config":
        from src.setup import _ask, _ask_int, _ask_yn, _ask_required, _verify_api_key

        print("\n  当前模型配置:")
        print(f"    模型名称: {config.model.name}")
        print(f"    接口地址: {config.model.base_url or '(未设置)'}")
        print(f"    API 密钥: {'***' if config.model.api_key else '(未设置)'}")
        print(f"    上下文窗口: {config.model.context_window}")
        print(f"    多模态: {'是' if config.model.multimodal else '否'}")
        print()

        # 模型名称
        if _ask_yn(f"  保留当前模型名称 ({config.model.name})？", default=True):
            pass
        else:
            config.model.name = _ask_required("  新模型名称", default=config.model.name)

        # 接口地址
        current_url = config.model.base_url or ""
        if _ask_yn(f"  保留当前接口地址 ({current_url or '(未设置)'})？", default=True):
            pass
        else:
            config.model.base_url = _ask("  新接口地址", default=current_url)

        # API 密钥
        if _ask_yn("  保留当前 API 密钥？", default=True):
            pass
        else:
            config.model.api_key = _ask_required("  新 API 密钥", default="")
            # 验证 API Key
            print("  验证 API Key 中...")
            success, msg = _verify_api_key(
                config.model.provider, config.model.name, config.model.base_url, config.model.api_key
            )
            if success:
                print("  [通过] API Key 验证成功")
            else:
                print(f"  [警告] API Key 验证失败: {msg}")
                if not _ask_yn("  是否仍然使用此 API Key？", default=True):
                    config.model.api_key = _ask_required("  重新输入 API 密钥", default="")

        # 上下文窗口
        if _ask_yn(f"  保留当前上下文窗口 ({config.model.context_window})？", default=True):
            pass
        else:
            config.model.context_window = _ask_int("  上下文窗口大小 (tokens)", default=config.model.context_window)

        # 多模态
        if _ask_yn(f"  保留当前多模态设置 ({'是' if config.model.multimodal else '否'})？", default=True):
            pass
        else:
            config.model.multimodal = _ask_yn("  该模型是否支持多模态（视觉输入）？", default=config.model.multimodal)

        # 保存配置
        from src.config import save_config

        save_config(config)
        print("\n  配置已保存")
        return 0

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flyclaw", description="flyclaw AI 助手")
    parser.add_argument("--version", action="version", version=f"flyclaw {__version__}")
    sub = parser.add_subparsers(dest="command", help="管理命令")

    sub.add_parser("doctor", help="运行系统诊断")
    sub.add_parser("status", help="显示系统状态")
    sub.add_parser("sessions", help="列出活跃会话")
    sub.add_parser("setup", help="交互式配置向导")

    # model 子命令
    model_parser = sub.add_parser("model", help="模型管理")
    model_sub = model_parser.add_subparsers(dest="model_command", help="模型命令")
    model_sub.add_parser("list", help="列出所有模型")
    model_sub.add_parser("test", help="测试当前模型")
    switch_parser = model_sub.add_parser("switch", help="切换模型")
    switch_parser.add_argument("id", type=int, help="模型 ID")
    add_parser = model_sub.add_parser("add", help="添加新模型")
    model_sub.add_parser("config", help="交互式配置当前模型")

    return parser


def cli_main():
    from src.instance import parse_instance_from_argv, set_instance

    n = parse_instance_from_argv()
    set_instance(n)

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
    elif args.command == "model":
        sys.exit(cmd_model(args))
    else:
        from src.main import main

        main()
