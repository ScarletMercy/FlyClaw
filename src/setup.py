"""Interactive configuration wizard for MyClaw.

Usage:
    python -m src.setup
    myclaw-setup
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


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return val or default


def _ask_choice(prompt: str, choices: list[str], default: str = "") -> str:
    options = ", ".join(choices)
    while True:
        val = _ask(f"{prompt} ({options})", default)
        if val in choices:
            return val
        print(f"  Please choose from: {options}")


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


def run_wizard():
    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║      MyClaw Configuration Wizard     ║")
    print("  ╚══════════════════════════════════════╝")
    print()

    existing = _load_existing()
    config = existing or {}

    # ── Model Provider ──
    print("  [1/5] Model Provider")
    print("  ────────────────────")
    provider_keys = list(PRESETS.keys())
    preset_names = {
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
    choices_display = [f"{k} ({preset_names.get(k, k)})" for k in provider_keys]
    current_provider = config.get("model", {}).get("provider", "anthropic")
    choice = _ask_choice("  Select provider", provider_keys, default=current_provider)

    preset = PRESETS[choice]
    model_section = config.get("model", {})

    if preset:
        model_section["provider"] = preset["provider"]
        model_section["name"] = _ask("  Model name", default=preset["name"])
        if preset["base_url"]:
            model_section["base_url"] = _ask("  Base URL", default=preset["base_url"])
        else:
            model_section.pop("base_url", None)
        if preset["env_key"]:
            env_name = preset["env_key"]
            has_key = bool(os.environ.get(env_name))
            model_section["api_key"] = _ask(f"  API key env var ({env_name})", default="${" + env_name + "}" if not has_key else "$" + env_name + " (set)")
        else:
            model_section.pop("api_key", None)
    else:
        # Custom provider
        model_section["provider"] = "openai"
        model_section["name"] = _ask("  Model name")
        model_section["base_url"] = _ask("  Base URL")
        env_name = _ask("  API key env var name", default="OPENAI_API_KEY")
        model_section["api_key"] = f"${{{env_name}}}"

    model_section["temperature"] = float(_ask("  Temperature", default=str(model_section.get("temperature", 0.0))))
    config["model"] = model_section

    # ── Gateway ──
    print()
    print("  [2/5] Gateway")
    print("  ────────────")
    gw = config.get("gateway", {})
    gw["host"] = _ask("  Listen host", default=gw.get("host", "127.0.0.1"))
    gw["port"] = int(_ask("  Listen port", default=str(gw.get("port", 18080))))
    token = _ask("  Auth token (leave empty for no auth)", default="")
    gw["auth_token"] = f"${{GATEWAY_AUTH_TOKEN}}" if not token else token
    config["gateway"] = gw

    # ── Feishu Channel ──
    print()
    print("  [3/5] Feishu Channel (optional)")
    print("  ─────────────────────────────────")
    enable_feishu = _ask_choice("  Enable Feishu?", ["yes", "no"], default="yes" if config.get("channels", {}).get("feishu", {}).get("enabled") else "no")
    feishu = config.get("channels", {}).get("feishu", {})
    feishu["enabled"] = enable_feishu == "yes"
    if feishu["enabled"]:
        feishu["app_id"] = _ask("  App ID", default=feishu.get("app_id", "${FEISHU_APP_ID}"))
        feishu["app_secret"] = _ask("  App Secret", default=feishu.get("app_secret", "${FEISHU_APP_SECRET}"))
        feishu["domain"] = _ask_choice("  Domain", ["feishu", "lark"], default=feishu.get("domain", "feishu"))
    channels = config.get("channels", {})
    channels["feishu"] = feishu
    config["channels"] = channels

    # ── Tavily (Search + Fetch) ──
    print()
    print("  [4/5] Web Search & Fetch (optional)")
    print("  ───────────────────────────────────")
    enable_search = _ask_choice("  Enable web search?", ["yes", "no"], default="yes" if config.get("tools", {}).get("web_search", {}).get("enabled") else "no")
    tools = config.get("tools", {})
    ws = tools.get("web_search", {})
    ws["enabled"] = enable_search == "yes"
    if ws["enabled"]:
        ws["api_key"] = _ask("  Tavily API key", default="${TAVILY_API_KEY}")
    tools["web_search"] = ws
    tools.setdefault("web_fetch", {})["enabled"] = ws["enabled"]
    config["tools"] = tools

    # ── Review ──
    print()
    print("  [5/5] Summary")
    print("  ────────────")
    print(f"  Model:        {model_section.get('provider')}/{model_section.get('name')}")
    if model_section.get("base_url"):
        print(f"  Base URL:     {model_section.get('base_url')}")
    print(f"  Gateway:      {gw['host']}:{gw['port']}")
    print(f"  Feishu:       {'enabled' if feishu['enabled'] else 'disabled'}")
    print(f"  Web search:   {'enabled' if ws['enabled'] else 'disabled'}")

    save = _ask_choice("  Save config?", ["yes", "no"], default="yes")
    if save == "yes":
        _save_config(config)
        print()
        print("  Next steps:")
        print("    1. Set required environment variables")
        print("    2. Run: python -m src.main")
        print()

    return config


def main():
    run_wizard()


if __name__ == "__main__":
    main()
