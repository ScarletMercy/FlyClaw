"""flyclaw 交互式配置向导。

用法:
    python -m src.setup
    flyclaw-setup
    flyclaw setup
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import yaml


def _get_config_path() -> Path:
    from src.instance import config_path

    return config_path()


PRESETS = {
    "openai": {
        "provider": "openai",
        "name": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
        "base_url": None,
    },
    "deepseek": {
        "provider": "openai",
        "name": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
    },
    "groq": {
        "provider": "openai",
        "name": "llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "together": {
        "provider": "openai",
        "name": "meta-llama/Llama-3-70b-chat-hf",
        "env_key": "TOGETHER_API_KEY",
        "base_url": "https://api.together.xyz/v1",
    },
    "ollama": {
        "provider": "openai",
        "name": "llama3",
        "env_key": "",
        "base_url": "http://localhost:11434/v1",
    },
    "zhipu": {
        "provider": "openai",
        "name": "glm-4-plus",
        "env_key": "ZHIPU_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
    },
    "moonshot": {
        "provider": "openai",
        "name": "moonshot-v1-128k",
        "env_key": "MOONSHOT_API_KEY",
        "base_url": "https://api.moonshot.cn/v1",
    },
    "qwen": {
        "provider": "openai",
        "name": "qwen-plus",
        "env_key": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "custom": None,
}

# ── Prompt helpers ──


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"  {prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return val or default


def _ask_required(prompt: str, default: str = "") -> str:
    """要求用户必须输入，不能为空。"""
    while True:
        suffix = f" [{default}]" if default else ""
        try:
            val = input(f"  {prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if val:
            return val
        if default:
            return default
        print("  [错误] 此项不能为空，请重新输入")


async def _verify_api_key_async(provider: str, name: str, base_url: str, api_key: str) -> tuple[bool, str]:
    """验证 API Key 是否可用。"""
    try:
        from src.agent.client import ChatClient

        client = ChatClient(base_url=base_url or "", api_key=api_key, model=name)
        resp = await client.chat_simple([{"role": "user", "content": "你好"}])
        return True, resp
    except Exception as e:
        return False, str(e)


def _verify_api_key(provider: str, name: str, base_url: str, api_key: str) -> tuple[bool, str]:
    """验证 API Key 是否可用（同步包装）。"""
    return asyncio.run(_verify_api_key_async(provider, name, base_url, api_key))


def _ask_choice(prompt: str, choices: list[str], default: str = "") -> str:
    options = ", ".join(choices)
    while True:
        val = _ask(prompt + f" ({options})", default)
        if val in choices:
            return val
        print(f"  请从以下选项中选择: {options}")


def _ask_yn(prompt: str, default: bool = True) -> bool:
    choices = "yes, no"
    default_str = "yes" if default else "no"
    return _ask_choice(prompt, ["yes", "no"], default_str) == "yes"


def _has_values(d: dict, *keys: str) -> bool:
    return all(d.get(k) for k in keys)


def _ask_skip(section_label: str, d: dict, *essential_keys: str) -> bool:
    if not _has_values(d, *essential_keys):
        return False
    print(f"  {section_label}已配置。")
    return _ask_yn(f"  保留当前{section_label}设置？", default=True)


# ── Config I/O ──


def _load_existing() -> dict | None:
    cfg_path = _get_config_path()
    if not cfg_path.exists():
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def _save_config(config: dict) -> None:
    cfg_path = _get_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"\n  配置已保存到 {cfg_path}")


def _section(config: dict, *keys: str) -> dict:
    d = config
    for k in keys:
        d = d.setdefault(k, {})
    return d


# ── Wizard steps ──


def _configure_fallbacks(model: dict) -> None:
    existing_fallbacks = model.get("fallbacks", [])
    print()
    if existing_fallbacks:
        print(f"  回退模型 ({len(existing_fallbacks)}):")
        for i, fb in enumerate(existing_fallbacks):
            print(f"    {i + 1}. {fb.get('provider', '?')}/{fb.get('name', '?')}")
    else:
        print("  未配置回退模型。")

    if not _ask_yn("  添加/编辑回退模型？", default=False):
        return

    fallbacks = list(existing_fallbacks)
    while True:
        print()
        if fallbacks:
            print(f"  当前回退模型 ({len(fallbacks)}):")
            for i, fb in enumerate(fallbacks):
                print(f"    {i + 1}. {fb.get('provider', '?')}/{fb.get('name', '?')}")
            action = _ask_choice("  操作", ["添加", "删除", "完成"], default="完成")
        else:
            action = "添加"

        if action == "完成":
            break
        elif action == "删除":
            idx = int(_ask(f"  删除编号 (1-{len(fallbacks)})", default="1")) - 1
            if 0 <= idx < len(fallbacks):
                removed = fallbacks.pop(idx)
                print(f"  已删除 {removed.get('provider')}/{removed.get('name')}")
            else:
                print("  无效编号。")
        elif action == "添加":
            choice = _ask_choice("  回退模型提供商", list(PRESETS.keys()), default="custom")
            preset = PRESETS[choice]
            fb = {}
            if preset:
                fb["provider"] = preset["provider"]
                fb["name"] = _ask("  模型名称", default=preset["name"])
                if preset["base_url"]:
                    fb["base_url"] = _ask("  接口地址", default=preset["base_url"])
                if preset["env_key"]:
                    env_name = preset["env_key"]
                    default_key = ""
                    fb["api_key"] = _ask(f"  API 密钥 ({env_name})", default=default_key)
            else:
                fb["provider"] = "openai"
                fb["name"] = _ask("  模型名称", default="")
                fb["base_url"] = _ask("  接口地址", default="")
                fb["api_key"] = _ask("  API 密钥", default="")
            fallbacks.append(fb)
            print(f"  已添加 {fb['provider']}/{fb['name']}")

    model["fallbacks"] = fallbacks


def _step_model(config: dict) -> None:
    print("  [1/8] 模型提供商")
    print("  ────────────────────")

    model = _section(config, "model")

    if _ask_skip("模型", model, "provider", "name", "api_key"):
        _configure_fallbacks(model)
        return
    existing_provider = model.get("provider", "openai")

    current_preset = "custom"
    for key, preset in PRESETS.items():
        if preset and preset["provider"] == existing_provider and preset.get("base_url") == model.get("base_url"):
            current_preset = key
            break

    choice = _ask_choice("  选择提供商", list(PRESETS.keys()), default="custom")
    preset = PRESETS[choice]

    if preset:
        model["provider"] = preset["provider"]
        model["name"] = _ask("  模型名称", default=model.get("name", preset["name"]))
        if preset["base_url"]:
            model["base_url"] = _ask("  接口地址", default=model.get("base_url", preset["base_url"]))
        else:
            model.pop("base_url", None)
        if preset["env_key"]:
            env_name = preset["env_key"]
            current_api_key = model.get("api_key", "")
            if current_api_key and current_api_key.startswith("${") and current_api_key.endswith("}"):
                inner = current_api_key[2:-1]
                if not inner.isupper() or "_" not in inner:
                    current_api_key = inner
            default_key = current_api_key or ""
            model["api_key"] = _ask_required(f"  API 密钥 ({env_name})", default=default_key)
            # 验证 API Key
            if model["api_key"]:
                print("  验证 API Key 中...")
                success, msg = _verify_api_key(
                    model.get("provider", ""), model.get("name", ""), model.get("base_url", ""), model["api_key"]
                )
                if success:
                    print("  [通过] API Key 验证成功")
                    ctx_default = str(model.get("context_window", 1000000))
                    model["context_window"] = int(_ask("  上下文窗口大小 (tokens)", default=ctx_default))
                    print(f"  上下文窗口: {model['context_window']} tokens")
                else:
                    print(f"  [警告] API Key 验证失败: {msg}")
                    if not _ask_yn("  是否仍然使用此 API Key？", default=True):
                        model["api_key"] = _ask_required(f"  重新输入 API 密钥 ({env_name})", default="")
        else:
            model.pop("api_key", None)
            ctx_default = str(model.get("context_window", 1000000))
            model["context_window"] = int(_ask("  上下文窗口大小 (tokens)", default=ctx_default))
            print(f"  上下文窗口: {model['context_window']} tokens")
    else:
        model["provider"] = "openai"
        model["name"] = _ask("  模型名称", default=model.get("name", ""))
        model["base_url"] = _ask("  接口地址", default=model.get("base_url", ""))
        model["api_key"] = _ask_required("  API 密钥", default=model.get("api_key", ""))
        # 验证 API Key
        if model["api_key"]:
            print("  验证 API Key 中...")
            success, msg = _verify_api_key(
                model.get("provider", ""), model.get("name", ""), model.get("base_url", ""), model["api_key"]
            )
            if success:
                print("  [通过] API Key 验证成功")
                ctx_default = str(model.get("context_window", 1000000))
                model["context_window"] = int(_ask("  上下文窗口大小 (tokens)", default=ctx_default))
                print(f"  上下文窗口: {model['context_window']} tokens")
            else:
                print(f"  [警告] API Key 验证失败: {msg}")
                if not _ask_yn("  是否仍然使用此 API Key？", default=True):
                    model["api_key"] = _ask_required("  重新输入 API 密钥", default="")

    model.setdefault("temperature", 1.0)

    _configure_fallbacks(model)


def _step_gateway(config: dict) -> None:
    print()
    print("  [2/8] 网关配置")
    print("  ────────────")

    gw = _section(config, "gateway")

    if _ask_skip("网关", gw, "port"):
        return
    # host 已固定为本地回环(代码常量),无需配置。
    # auth_token 留空——首次启动时自动生成强令牌,无需在此配置。
    from src.instance import get_instance

    default_port = 18080 + (get_instance() or 0)
    gw["port"] = int(_ask("  监听端口", default=str(gw.get("port", default_port))))


def _step_channel(config: dict) -> None:
    print()
    print("  [3/8] 渠道配置")
    print("  ─────────────")

    channels = _section(config, "channels")
    qq = _section(channels, "qq")
    weixin = _section(channels, "weixin")

    # Determine current default based on existing config
    if weixin.get("enabled") and not qq.get("enabled"):
        default_channel = "微信"
    else:
        default_channel = "QQ"

    channel = _ask_choice("  选择渠道", ["QQ", "微信"], default=default_channel)

    if channel == "QQ":
        weixin["enabled"] = False
        _configure_qq(qq)
    else:
        qq["enabled"] = False
        _configure_weixin(weixin)


def _configure_qq(qq: dict) -> None:
    """Configure QQ channel (called from _step_channel)."""
    qq["enabled"] = True

    if qq.get("app_id") and qq.get("client_secret") and _ask_skip("QQ 机器人", qq, "app_id", "client_secret"):
        return

    has_existing = bool(qq.get("app_id") and qq.get("client_secret"))
    if has_existing:
        method = _ask_choice("  配置方式", ["保留当前配置", "扫码一键配置", "手动输入"], default="保留当前配置")
    else:
        method = _ask_choice("  配置方式", ["扫码一键配置", "手动输入"], default="扫码一键配置")

    if method == "保留当前配置":
        pass
    elif method == "扫码一键配置":
        print()
        print("  正在生成配置链接...")
        try:
            import cryptography  # noqa: F401
        except ImportError:
            print("  缺少 cryptography 库，请运行: pip install cryptography")
            result = None
        else:
            try:
                from src.channels.qq_onboard import qr_register

                result = qr_register()
            except Exception as exc:
                print(f"  扫码配置失败: {exc}")
                result = None

        if result:
            qq["app_id"] = result["app_id"]
            qq["client_secret"] = result["client_secret"]
            print(f"  已自动填入 App ID: {result['app_id']}")
            if result.get("user_openid") and not qq.get("allow_from"):
                qq.setdefault("allow_from", [result["user_openid"]])
        else:
            print("  扫码未完成，将使用手动输入。")
            qq["app_id"] = _ask("  应用 ID (App ID)", default=qq.get("app_id", ""))
            qq["client_secret"] = _ask("  应用密钥 (Client Secret)", default=qq.get("client_secret", ""))
    else:
        qq["app_id"] = _ask("  应用 ID (App ID)", default=qq.get("app_id", ""))
        qq["client_secret"] = _ask("  应用密钥 (Client Secret)", default=qq.get("client_secret", ""))

    qq["dm_policy"] = _ask_choice("  私聊策略", ["open", "allowlist"], default=qq.get("dm_policy", "open"))
    qq["group_policy"] = _ask_choice(
        "  群聊策略", ["allowlist", "open", "disabled"], default=qq.get("group_policy", "allowlist")
    )
    qq["require_mention"] = _ask_yn("  群聊中需要 @机器人？", default=qq.get("require_mention", True))
    qq["markdown_support"] = _ask_yn("  启用 Markdown 支持？", default=qq.get("markdown_support", True))


def _configure_weixin(weixin: dict) -> None:
    """Configure WeChat channel (called from _step_channel)."""
    weixin["enabled"] = True

    if weixin.get("account_id") and weixin.get("token") and _ask_skip("微信机器人", weixin, "account_id", "token"):
        return

    has_existing = bool(weixin.get("account_id") and weixin.get("token"))
    if has_existing:
        method = _ask_choice("  配置方式", ["保留当前配置", "扫码登录", "手动输入"], default="保留当前配置")
    else:
        method = _ask_choice("  配置方式", ["扫码登录", "手动输入"], default="扫码登录")

    if method == "保留当前配置":
        pass
    elif method == "扫码登录":
        print()
        print("  正在获取微信二维码...")
        try:
            from src.channels.weixin_onboard import qr_login

            result = qr_login()
        except ImportError:
            print("  缺少 aiohttp 或 cryptography 库，请运行: pip install aiohttp cryptography")
            result = None
        except Exception as exc:
            print(f"  扫码登录失败: {exc}")
            result = None

        if result:
            weixin["account_id"] = result["account_id"]
            weixin["token"] = result["token"]
            weixin["base_url"] = result.get("base_url", "https://ilinkai.weixin.qq.com")
            print(f"  已自动填入 Account ID: {result['account_id']}")
            if result.get("user_id") and not weixin.get("allowed_users"):
                weixin.setdefault("allowed_users", [result["user_id"]])
        else:
            print("  扫码未完成，将使用手动输入。")
            weixin["account_id"] = _ask("  Account ID", default=weixin.get("account_id", ""))
            weixin["token"] = _ask("  Token", default=weixin.get("token", ""))
    else:
        weixin["account_id"] = _ask("  Account ID", default=weixin.get("account_id", ""))
        weixin["token"] = _ask("  Token", default=weixin.get("token", ""))

    weixin["dm_policy"] = _ask_choice(
        "  私聊策略", ["open", "allowlist", "disabled"], default=weixin.get("dm_policy", "open")
    )
    weixin["group_policy"] = _ask_choice(
        "  群聊策略", ["disabled", "allowlist", "open"], default=weixin.get("group_policy", "disabled")
    )


def _step_search(config: dict) -> None:
    print()
    print("  [4/8] 网页搜索")
    print("  ────────────────")

    tools = _section(config, "tools")
    ws = _section(tools, "web_search")

    if ws.get("enabled") and _ask_skip("网页搜索", ws, "api_key"):
        return

    ws["enabled"] = _ask_yn("  启用网页搜索？", default=ws.get("enabled", False))
    if ws["enabled"]:
        ws["api_key"] = _ask("  Tavily API 密钥", default=ws.get("api_key", "${TAVILY_API_KEY}"))


def _check_chromium_installed() -> bool:
    """检查 Playwright Chromium 是否已安装。"""
    import subprocess

    try:
        ret = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "--list"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        for line in ret.stdout.splitlines():
            stripped = line.strip().lower()
            if "chromium-" in stripped or stripped.endswith("chromium"):
                return True
    except Exception:
        pass
    return False


def _step_browser(config: dict) -> None:
    print()
    print("  [5/8] 浏览器工具（可选）")
    print("  ────────────────────────────")
    print("  启用后，AI 可以通过浏览器访问网页、截图、执行自动化操作。")

    tools = _section(config, "tools")
    br = _section(tools, "browser")

    enabled = _ask_yn("  启用浏览器工具？", default=br.get("enabled", False))
    br["enabled"] = enabled

    if enabled:
        if _check_chromium_installed():
            print("  Chromium 已安装，无需重复安装。")
        elif _ask_yn("  现在安装 Chromium 浏览器（Playwright）？", default=True):
            print("  正在安装 Chromium...")
            import subprocess

            # 镜像源列表（按优先级排序），逐个尝试直到成功
            mirrors = [
                ("淘宝 NPM 镜像", "https://registry.npmmirror.com/-/binary/playwright"),
                ("Playwright 官方 CDN", "https://playwright.azureedge.net"),
            ]

            installed = False
            for name, host in mirrors:
                env = os.environ.copy()
                env["PLAYWRIGHT_DOWNLOAD_HOST"] = host
                print(f"  尝试使用 {name} ({host}) ...")
                ret = subprocess.run(
                    [sys.executable, "-m", "playwright", "install", "chromium"],
                    capture_output=False,
                    env=env,
                )
                if ret.returncode == 0:
                    print(f"  Chromium 安装完成（使用 {name}）。")
                    installed = True
                    break
                else:
                    print(f"  {name} 安装失败，尝试下一个镜像源...")

            if not installed:
                print("  所有镜像源均安装失败，请稍后手动运行: playwright install chromium")


def _step_media_understanding(config: dict) -> None:
    print()
    print("  [6/8] 媒体理解（可选）")
    print("  ────────────────────────────")
    print("  启用后，AI 可以识别图片内容、转录音频等。")
    print("  需要多模态模型 API（如 OpenAI GPT-4o 等）。")

    tools = _section(config, "tools")
    mu = _section(tools, "media_understanding")

    enabled = _ask_yn("  启用媒体理解？", default=mu.get("enabled", False))
    mu["enabled"] = enabled

    if enabled:
        mu["provider"] = _ask("  模型提供商", default=mu.get("provider", "openai"))
        mu["name"] = _ask("  模型名称", default=mu.get("name", "gpt-4o-mini"))
        mu["base_url"] = _ask("  接口地址（留空使用默认）", default=mu.get("base_url", ""))
        mu["api_key"] = _ask("  API 密钥", default=mu.get("api_key", "${OPENAI_API_KEY}"))

        # 验证 API Key
        if mu["api_key"]:
            print("  验证 API Key 中...")
            success, msg = _verify_api_key(
                mu.get("provider", ""), mu.get("name", ""), mu.get("base_url", ""), mu["api_key"]
            )
            if success:
                print("  [通过] API Key 验证成功")
            else:
                print(f"  [警告] API Key 验证失败: {msg}")
                if not _ask_yn("  是否仍然使用此 API Key？", default=True):
                    mu["api_key"] = _ask("  重新输入 API 密钥", default="")


def _step_memory_store(config: dict) -> None:
    print()
    print("  [7/8] 记忆存储（可选）")
    print("  ─────────────────────────")
    print("  启用后，AI 可以记住你的偏好、身份等信息。")
    print("  记忆数据保存在本地，启用后自动从对话中提取并保存要点。")

    ms = _section(config, "memory_store")

    enabled = _ask_yn("  启用记忆存储？", default=ms.get("enabled", True))
    ms["enabled"] = enabled


def _step_summary(config: dict) -> None:
    model = config.get("model", {})
    gw = config.get("gateway", {})
    qq = config.get("channels", {}).get("qq", {})
    weixin = config.get("channels", {}).get("weixin", {})
    ws = config.get("tools", {}).get("web_search", {})
    br = config.get("tools", {}).get("browser", {})
    mu = config.get("tools", {}).get("media_understanding", {})
    ms = config.get("memory_store", {})

    # host 已固化为代码常量(不再写 config.yaml),这里直接取,避免显示成 "?"
    from src.gateway import GATEWAY_HOST

    print()
    print("  [8/8] 配置总览")
    print("  ─────────────")
    print(f"  模型:       {model.get('provider', '?')}/{model.get('name', '?')}")
    print(f"  上下文窗口: {model.get('context_window', 1000000)} tokens")
    if model.get("base_url"):
        print(f"  接口地址:   {model['base_url']}")
    fallbacks = model.get("fallbacks", [])
    if fallbacks:
        print(f"  回退模型:   {len(fallbacks)}")
        for fb in fallbacks:
            print(f"    - {fb.get('provider', '?')}/{fb.get('name', '?')}")
    print(f"  网关:       {GATEWAY_HOST}:{gw.get('port', '?')}")
    if qq.get("enabled"):
        print(f"  渠道:       QQ (app_id: {qq.get('app_id', '')})")
    elif weixin.get("enabled"):
        print(f"  渠道:       微信 (account_id: {weixin.get('account_id', '')})")
    else:
        print(f"  渠道:       未启用")
    print(f"  网页搜索:   {'已启用' if ws.get('enabled') else '未启用'}")
    print(f"  浏览器工具: {'已启用' if br.get('enabled') else '未启用'}")
    print(f"  媒体理解:   {'已启用' if mu.get('enabled') else '未启用'}")
    if mu.get("enabled") and mu.get("name"):
        print(f"    模型:     {mu['name']}")
    print(f"  记忆存储:   {'已启用' if ms.get('enabled') else '未启用'}")


# ── Main ──


def run_wizard():
    from src.instance import get_instance, instance_label

    label = instance_label()
    print()
    if label:
        print(f"  ╔══════════════════════════════════════╗")
        print(f"  ║    flyclaw 配置向导 — 实例 {get_instance()}        ║")
        print(f"  ╚══════════════════════════════════════╝")
    else:
        print("  ╔══════════════════════════════════════╗")
        print("  ║        flyclaw 配置向导               ║")
        print("  ╚══════════════════════════════════════╝")
    print()
    print("  按 Enter 保留当前/默认值，按 Ctrl+C 退出且不保存。")
    print()

    existing = _load_existing()
    config = existing or {}

    _step_model(config)
    _step_gateway(config)
    _step_channel(config)
    _step_search(config)
    _step_browser(config)
    _step_media_understanding(config)
    _step_memory_store(config)
    _step_summary(config)

    save = _ask_yn("  保存配置？")
    if save:
        _save_config(config)
        print()
        print("  后续步骤:")
        print("    1. 运行: flyclaw")
        print("    2. 或运行: flyclaw doctor（检查配置）")
        print()
    else:
        print("\n  配置已丢弃。")


def main():
    from src.instance import parse_instance_from_argv, set_instance

    n = parse_instance_from_argv()
    set_instance(n)
    run_wizard()


if __name__ == "__main__":
    main()
