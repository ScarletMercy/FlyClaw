"""Interactive configuration wizard for MyClaw.

Usage:
    python -m src.setup
    myclaw-setup
    myclaw setup
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

CONFIG_PATH = Path("config.yaml")

PRESETS = {
    "anthropic": {
        "provider": "anthropic",
        "name": "claude-sonnet-4-6",
        "env_key": "ANTHROPIC_API_KEY",
        "base_url": None,
    },
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

PRESET_LABELS = {
    "anthropic": "Anthropic (Claude)",
    "openai": "OpenAI (GPT)",
    "deepseek": "DeepSeek",
    "groq": "Groq (Llama)",
    "together": "Together AI (Llama)",
    "ollama": "Ollama (Local)",
    "zhipu": "ZhiPu (GLM)",
    "moonshot": "Moonshot (Kimi)",
    "qwen": "Qwen (Alibaba)",
    "custom": "Custom OpenAI-compatible",
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


def _ask_choice(prompt: str, choices: list[str], default: str = "") -> str:
    options = ", ".join(choices)
    while True:
        val = _ask(prompt + f" ({options})", default)
        if val in choices:
            return val
        print(f"  Please choose from: {options}")


def _ask_yn(prompt: str, default: bool = True) -> bool:
    choices = "yes, no"
    default_str = "yes" if default else "no"
    return _ask_choice(prompt, ["yes", "no"], default_str) == "yes"


def _has_values(d: dict, *keys: str) -> bool:
    """Check if a dict has non-empty values for all given keys."""
    return all(d.get(k) for k in keys)


def _ask_skip(section_label: str, d: dict, *essential_keys: str) -> bool:
    """Ask user whether to skip reconfiguring a section that already has values.

    Returns True if the user wants to skip (keep existing config).
    """
    if not _has_values(d, *essential_keys):
        return False
    print(f"  {section_label} already configured.")
    return _ask_yn(f"  Keep current {section_label} settings?", default=True)


# ── Config I/O ──


def _load_existing() -> dict | None:
    if not CONFIG_PATH.exists():
        return None
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def _save_config(config: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"\n  Config saved to {CONFIG_PATH}")


def _section(config: dict, *keys: str) -> dict:
    d = config
    for k in keys:
        d = d.setdefault(k, {})
    return d


# ── Wizard steps ──


def _configure_fallbacks(model: dict) -> None:
    """Configure fallback models (can be called independently of main model config)."""
    existing_fallbacks = model.get("fallbacks", [])
    print()
    if existing_fallbacks:
        print(f"  Fallback models ({len(existing_fallbacks)}):")
        for i, fb in enumerate(existing_fallbacks):
            print(f"    {i + 1}. {fb.get('provider', '?')}/{fb.get('name', '?')}")
    else:
        print("  No fallback models configured.")

    if not _ask_yn("  Add/edit fallback models?", default=False):
        return

    fallbacks = list(existing_fallbacks)
    while True:
        print()
        if fallbacks:
            print(f"  Current fallbacks ({len(fallbacks)}):")
            for i, fb in enumerate(fallbacks):
                print(f"    {i + 1}. {fb.get('provider', '?')}/{fb.get('name', '?')}")
            action = _ask_choice("  Action", ["add", "remove", "done"], default="done")
        else:
            action = "add"

        if action == "done":
            break
        elif action == "remove":
            idx = int(_ask(f"  Remove number (1-{len(fallbacks)})", default="1")) - 1
            if 0 <= idx < len(fallbacks):
                removed = fallbacks.pop(idx)
                print(f"  Removed {removed.get('provider')}/{removed.get('name')}")
            else:
                print("  Invalid number.")
        elif action == "add":
            choice = _ask_choice("  Fallback provider", list(PRESETS.keys()), default="custom")
            preset = PRESETS[choice]
            fb = {}
            if preset:
                fb["provider"] = preset["provider"]
                fb["name"] = _ask("  Model name", default=preset["name"])
                if preset["base_url"]:
                    fb["base_url"] = _ask("  Base URL", default=preset["base_url"])
                if preset["env_key"]:
                    env_name = preset["env_key"]
                    default_key = f"${{{env_name}}}"
                    fb["api_key"] = _ask(f"  API key ({env_name})", default=default_key)
            else:
                fb["provider"] = "openai"
                fb["name"] = _ask("  Model name", default="")
                fb["base_url"] = _ask("  Base URL", default="")
                env_name = _ask("  API key env var name", default="OPENAI_API_KEY")
                fb["api_key"] = f"${{{env_name}}}"
            fallbacks.append(fb)
            print(f"  Added {fb['provider']}/{fb['name']}")

    model["fallbacks"] = fallbacks


def _step_model(config: dict) -> None:
    print("  [1/6] Model Provider")
    print("  ────────────────────")

    model = _section(config, "model")

    if _ask_skip("Model", model, "provider", "name", "api_key"):
        _configure_fallbacks(model)
        return
    existing_provider = model.get("provider", "anthropic")

    # Detect current preset
    current_preset = "custom"
    for key, preset in PRESETS.items():
        if preset and preset["provider"] == existing_provider and preset.get("base_url") == model.get("base_url"):
            current_preset = key
            break

    choice = _ask_choice("  Select provider", list(PRESETS.keys()), default=current_preset)
    preset = PRESETS[choice]

    if preset:
        model["provider"] = preset["provider"]
        model["name"] = _ask("  Model name", default=model.get("name", preset["name"]))
        if preset["base_url"]:
            model["base_url"] = _ask("  Base URL", default=model.get("base_url", preset["base_url"]))
        else:
            model.pop("base_url", None)
        if preset["env_key"]:
            env_name = preset["env_key"]
            has_key = bool(os.environ.get(env_name))
            current_api_key = model.get("api_key", "")
            if current_api_key and current_api_key.startswith("${") and current_api_key.endswith("}"):
                inner = current_api_key[2:-1]
                if not inner.isupper() or "_" not in inner:
                    current_api_key = inner
            if current_api_key:
                default_key = current_api_key
            elif has_key:
                default_key = f"${{{env_name}}} (set in env)"
            else:
                default_key = f"${{{env_name}}}"
            model["api_key"] = _ask(f"  API key ({env_name})", default=default_key)
        else:
            model.pop("api_key", None)
    else:
        model["provider"] = "openai"
        model["name"] = _ask("  Model name", default=model.get("name", ""))
        model["base_url"] = _ask("  Base URL", default=model.get("base_url", ""))
        env_name = _ask("  API key env var name", default="OPENAI_API_KEY")
        model["api_key"] = f"${{{env_name}}}"

    model["temperature"] = float(_ask("  Temperature", default=str(model.get("temperature", 0.0))))

    _configure_fallbacks(model)


def _step_gateway(config: dict) -> None:
    print()
    print("  [2/6] Gateway")
    print("  ────────────")

    gw = _section(config, "gateway")

    if _ask_skip("Gateway", gw, "host", "port"):
        return
    gw["host"] = _ask("  Listen host", default=gw.get("host", "127.0.0.1"))
    gw["port"] = int(_ask("  Listen port", default=str(gw.get("port", 18080))))
    token = _ask("  Auth token (leave empty for no auth)", default="")
    gw["auth_token"] = f"${{GATEWAY_AUTH_TOKEN}}" if not token else token


def _step_feishu(config: dict) -> None:
    print()
    print("  [3/6] Feishu Channel")
    print("  ────────────────────")

    channels = _section(config, "channels")
    feishu = _section(channels, "feishu")

    if feishu.get("enabled") and _ask_skip("Feishu", feishu, "app_id", "app_secret"):
        return

    enabled = _ask_yn("  Enable Feishu?", default=feishu.get("enabled", False))
    feishu["enabled"] = enabled

    if enabled:
        feishu["app_id"] = _ask("  App ID", default=feishu.get("app_id", ""))
        feishu["app_secret"] = _ask("  App Secret", default=feishu.get("app_secret", ""))
        feishu["domain"] = _ask_choice("  Domain", ["feishu", "lark"], default=feishu.get("domain", "feishu"))
        feishu["dm_policy"] = _ask_choice(
            "  DM policy", ["open", "pairing", "allowlist"], default=feishu.get("dm_policy", "open")
        )
        feishu["group_policy"] = _ask_choice(
            "  Group policy", ["allowlist", "open", "disabled"], default=feishu.get("group_policy", "allowlist")
        )
        feishu["require_mention"] = _ask_yn(
            "  Require @mention in groups?", default=feishu.get("require_mention", True)
        )


def _step_qq(config: dict) -> None:
    print()
    print("  [4/6] QQ Bot Channel")
    print("  ─────────────────────")

    channels = _section(config, "channels")
    qq = _section(channels, "qq")

    if qq.get("enabled") and _ask_skip("QQ Bot", qq, "app_id", "client_secret"):
        return

    enabled = _ask_yn("  Enable QQ Bot?", default=qq.get("enabled", False))
    qq["enabled"] = enabled

    if enabled:
        qq["app_id"] = _ask("  App ID", default=qq.get("app_id", ""))
        qq["client_secret"] = _ask("  Client Secret", default=qq.get("client_secret", ""))
        qq["dm_policy"] = _ask_choice("  DM policy", ["open", "allowlist"], default=qq.get("dm_policy", "open"))
        qq["group_policy"] = _ask_choice(
            "  Group policy", ["allowlist", "open", "disabled"], default=qq.get("group_policy", "allowlist")
        )
        qq["require_mention"] = _ask_yn("  Require @mention in groups?", default=qq.get("require_mention", True))
        qq["markdown_support"] = _ask_yn("  Enable markdown support?", default=qq.get("markdown_support", False))


def _step_search(config: dict) -> None:
    print()
    print("  [5/6] Web Search")
    print("  ────────────────")

    tools = _section(config, "tools")
    ws = _section(tools, "web_search")

    if ws.get("enabled") and _ask_skip("Web Search", ws, "api_key"):
        return

    ws["enabled"] = _ask_yn("  Enable web search?", default=ws.get("enabled", False))
    if ws["enabled"]:
        ws["api_key"] = _ask("  Tavily API key", default=ws.get("api_key", "${TAVILY_API_KEY}"))

    wf = _section(tools, "web_fetch")
    wf["enabled"] = ws["enabled"] if not wf.get("enabled") else True


def _step_summary(config: dict) -> None:
    model = config.get("model", {})
    gw = config.get("gateway", {})
    feishu = config.get("channels", {}).get("feishu", {})
    qq = config.get("channels", {}).get("qq", {})
    ws = config.get("tools", {}).get("web_search", {})

    print()
    print("  [6/6] Summary")
    print("  ─────────────")
    print(f"  Model:      {model.get('provider', '?')}/{model.get('name', '?')}")
    if model.get("base_url"):
        print(f"  Base URL:   {model['base_url']}")
    fallbacks = model.get("fallbacks", [])
    if fallbacks:
        print(f"  Fallbacks:  {len(fallbacks)}")
        for fb in fallbacks:
            print(f"    - {fb.get('provider', '?')}/{fb.get('name', '?')}")
    print(f"  Gateway:    {gw.get('host', '?')}:{gw.get('port', '?')}")
    print(f"  Feishu:     {'enabled' if feishu.get('enabled') else 'disabled'}")
    if feishu.get("enabled"):
        print(f"    domain:   {feishu.get('domain', 'feishu')}")
        print(f"    dm:       {feishu.get('dm_policy', 'open')}")
    print(f"  QQ Bot:     {'enabled' if qq.get('enabled') else 'disabled'}")
    if qq.get("enabled"):
        print(f"    app_id:   {qq.get('app_id', '')}")
    print(f"  Web search: {'enabled' if ws.get('enabled') else 'disabled'}")


# ── Main ──


def run_wizard():
    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║      MyClaw Configuration Wizard     ║")
    print("  ╚══════════════════════════════════════╝")
    print()
    print("  Press Enter at any prompt to keep the current/default value.")
    print("  Press Ctrl+C to exit without saving.")
    print()

    existing = _load_existing()
    config = existing or {}

    _step_model(config)
    _step_gateway(config)
    _step_feishu(config)
    _step_qq(config)
    _step_search(config)
    _step_summary(config)

    save = _ask_yn("  Save configuration?")
    if save:
        _save_config(config)
        print()
        print("  Next steps:")
        print("    1. Set required environment variables (API keys)")
        print("    2. Run: myclaw")
        print("    3. Or run: myclaw doctor  (to check configuration)")
        print()
    else:
        print("\n  Configuration discarded.")


def main():
    run_wizard()


if __name__ == "__main__":
    main()
