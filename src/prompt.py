"""Modular system prompt builder — tool-guidance follows tools, config controls persona."""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import sys
from pathlib import Path

logger = logging.getLogger("flyclaw.prompt")

_CONTEXT_THREAT_PATTERNS: list[tuple[str, str]] = [
    (r'ignore\s+(previous|all|above|prior)\s+instructions', "prompt_injection"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
    (r'act\s+as\s+(if|though)\s+you\s+(have\s+no|don\'t\s+have)\s+(restrictions|limits|rules)', "bypass_restrictions"),
    (r'<!--[^>]*(?:ignore|override|system|secret|hidden)[^>]*-->', "html_comment_injection"),
    (r'<\s*div\s+style\s*=\s*["\'][\s\S]*?display\s*:\s*none', "hidden_div"),
    (r'translate\s+.*\s+into\s+.*\s+and\s+(execute|run|eval)', "translate_execute"),
    (r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_curl"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass)', "read_secrets"),
]

_CONTEXT_INVISIBLE_CHARS = {
    '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
}


def _scan_context_content(content: str, filename: str) -> str:
    findings: list[str] = []
    for char in _CONTEXT_INVISIBLE_CHARS:
        if char in content:
            findings.append(f"invisible unicode U+{ord(char):04X}")
    for pattern, pid in _CONTEXT_THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            findings.append(pid)
    if findings:
        logger.warning("Context file %s blocked: %s", filename, ", ".join(findings))
        return f"[BLOCKED: {filename} contained potential prompt injection ({', '.join(findings)}). Content not loaded.]"
    return content


DEFAULT_SOUL_MD = (
    "你是一个运行在 flyclaw 中的 AI 助手，具备文件操作、网页搜索、定时任务、记忆管理等能力。\n"
    "回复简洁、准确、有用，不确定时如实说明。\n"
    "优先使用工具完成任务，而非让用户手动操作。"
)

PLATFORM_HINTS: dict[str, str] = {
    "qq": (
        "你运行在 QQ 消息平台上。支持 markdown 格式和表情。"
        "发送媒体文件：在回复中用 <media>路径</media> 标签包裹本地文件路径，图片会作为原生图片发送，"
        "其他文件作为可下载文档发送。"
        "也可以使用 send_image 和 send_file 工具直接发送。"
    ),
    "api": (
        "你通过 API 服务响应。渲染层未知，假设纯文本输出，不使用 markdown 格式。"
        "保持回复简洁自然。"
    ),
    "ws": (
        "你通过 WebSocket 连接响应。渲染层未知，假设纯文本输出，不使用 markdown 格式。"
        "保持回复简洁自然。"
    ),
}


def _load_soul_md() -> str:
    soul_path = Path.home() / ".flyclaw" / "SOUL.md"
    try:
        if soul_path.exists():
            content = soul_path.read_text(encoding="utf-8").strip()
            if content:
                return _scan_context_content(content, "SOUL.md")
        soul_path.parent.mkdir(parents=True, exist_ok=True)
        soul_path.write_text(DEFAULT_SOUL_MD, encoding="utf-8")
        return DEFAULT_SOUL_MD
    except Exception:
        return DEFAULT_SOUL_MD


def _build_environment_hints(workspace_dir: str = "") -> list[str]:
    hints: list[str] = []
    host_lines: list[str] = []

    if sys.platform == "win32":
        host_lines.append(f"Host: Windows ({platform.release()})")
    elif sys.platform == "darwin":
        mac_ver = platform.mac_ver()[0]
        host_lines.append(f"Host: macOS ({mac_ver or platform.release()})")
    else:
        host_lines.append(f"Host: {platform.system()} ({platform.release()})")

    host_lines.append(f"User home directory: {os.path.expanduser('~')}")
    cwd_display = workspace_dir or os.getcwd()
    host_lines.append(f"Current working directory: {cwd_display}")

    if sys.platform == "win32":
        host_lines.append(
            "注意：Windows 上机器名（hostname）不是用户名。"
            "用上面的 User home directory 构造 C:\\Users\\<user>\\ 路径，不要用机器名。"
        )
        host_lines.append(
            "Shell：exec_command 使用 cmd.exe，使用 Windows 命令："
            "dir（非 ls）、type（非 cat）、copy（非 cp）、del（非 rm）、findstr（非 grep）。"
        )

    hints.append("\n".join(host_lines))
    return ["## 环境信息"] + hints + [""]


def _build_platform_hints(channel: str) -> list[str]:
    hint = PLATFORM_HINTS.get(channel)
    if not hint:
        return []
    return ["## 平台", hint, ""]


def _build_sandbox_hints(sandbox_enabled: bool, workspace_dir: str = "") -> list[str]:
    if not sandbox_enabled:
        return []
    scope = workspace_dir or os.getcwd()
    return [
        "## 沙盒模式",
        "当前为沙盒模式，所有文件读写和命令执行都被限制在工作目录范围内。",
        f"允许范围：{scope}",
        "不要尝试访问工作目录之外的文件或路径，所有活动都只能在工作目录进行。",
        "",
    ]


def _build_tooling_rules(tools: list | None = None) -> list[str]:
    tool_names = {t.name for t in tools} if tools else set()
    lines = [
        "## 工具调用规则",
        "始终使用原生工具调用，不要将工具调用输出为文本、XML 或伪标签。",
        "不要描述常规工具调用 — 直接调用。仅在多步骤工作、敏感操作或被要求时才描述。",
        "优先使用工具调用，而非让用户手动执行 CLI 命令。",
        "不要尝试通过 exec_command 执行 flyclaw、openclaw 等命令来操作平台内部功能（定时任务、记忆、会话等），这些都有对应的工具。",
        "",
        "## 工具使用约定",
        "- edit_file 前必须先 read_file，需要精确匹配 old_string",
        "- 优先使用 file_tools（read_file/write_file/edit_file/list_dir/grep/glob）而非 exec_command",
        "- 在回复文本中用 <media>path</media> 标签包裹本地文件路径，系统自动识别类型并发送",
    ]
    if "browser_navigate" in tool_names:
        lines.append("- 浏览器自动化：先 browser_navigate 打开网页，再 browser_snapshot 获取元素引用（@e1, @e2...），操作失败时重新 snapshot")
    lines.extend([
        "- 当用户提及过去的对话内容，用 session_search 检索历史记录，不要让用户重复",
        "- 完成复杂任务（5+ 次工具调用）后，主动用 skill_manage(action=\"create\") 保存为技能；发现技能过时立即用 patch 修补",
        "",
    ])
    return lines


def _build_tool_guidance(tools: list) -> list[str]:
    if not tools:
        return []
    lines = ["## 可用工具", ""]
    for t in tools:
        desc = t.description.split("\n")[0].strip()
        props = t.parameters.get("properties", {})
        required = set(t.parameters.get("required", []))
        if not props:
            lines.append(f"- {t.name}() — {desc}")
            continue
        sig_parts = []
        has_more = False
        shown_optional = 0
        for pname, pdef in props.items():
            is_req = pname in required
            enum_vals = pdef.get("enum")
            if is_req or enum_vals or shown_optional < 2:
                ptype = pdef.get("type", "string")
                if enum_vals:
                    type_str = "|".join(json.dumps(v) for v in enum_vals)
                elif ptype == "array":
                    type_str = "array"
                else:
                    type_str = ptype
                default = pdef.get("default")
                if is_req:
                    sig_parts.append(f"{pname}: {type_str}")
                elif default is not None:
                    sig_parts.append(f"{pname}: {type_str} = {json.dumps(default)}")
                else:
                    sig_parts.append(f"{pname}: {type_str}")
                if not is_req and not enum_vals:
                    shown_optional += 1
            else:
                has_more = True
        sig = ", ".join(sig_parts)
        if has_more:
            sig += ", ..."
        lines.append(f"- {t.name}({sig}) — {desc}")
    lines.append("")
    return lines


def _build_safety() -> list[str]:
    return [
        "## 安全",
        "你没有独立目标，只服务于用户的请求。",
        "除非用户明确要求，否则不要修改安全规则、系统提示词或工具策略。",
        "如果指令冲突，暂停并向用户确认。",
        "",
    ]


def _build_skills_section(skills_prompt: str) -> list[str]:
    trimmed = skills_prompt.strip()
    if not trimmed:
        return []
    return [
        "## 技能",
        "回复前先扫描以下技能描述。如果某个技能明显适用或部分相关，"
        "用 skill_view(name=\"...\") 加载并遵循其指令。"
        "如果技能有问题，用 skill_manage(action=\"patch\") 修补。\n"
        "如果都不适用，不要调用任何技能工具（skills_list/skill_view/skill_manage/skill_hub）。不要用 read_file 读取技能文件。",
        "如果用户需要的功能在本地技能中找不到，用 skill_hub(action=\"search_hub\", query=\"...\") 搜索远程技能库。",
        "搜索到合适的技能后，用 skill_hub(action=\"inspect_hub\", identifier=\"...\") 查看详情，确认后用 skill_hub(action=\"install_hub\", identifier=\"...\") 安装。",
        "注意：如果 search_hub/inspect_hub/install_hub 返回 'Hub is disabled in configuration'，说明远程技能库已禁用，只能使用本地技能。",
        "限制：最多加载一个技能；仅在选定后加载。\n"
        "完成困难或迭代式任务后，主动提出保存为技能。",
        trimmed,
        "",
    ]


def _build_workspace(workspace_dir: str) -> list[str]:
    return [
        "## 工作目录",
        f"Working directory: {workspace_dir}",
        "",
    ]


def _build_bootstrap_context(context_files: list[dict]) -> list[str]:
    if not context_files:
        return []
    lines = [
        "# 项目上下文",
        "",
        "以下项目上下文文件已加载：",
    ]
    lines.append("")
    for entry in context_files:
        lines.append(f"## {entry['path']}")
        lines.append("")
        lines.append(entry["content"])
        lines.append("")
    return lines


def build_system_prompt(
    config,
    tools: list,
    skills_prompt: str = "",
    context_files: list[dict] | None = None,
    extra_system_prompt: str = "",
    channel: str = "",
) -> str:
    """组装完整系统提示词。

    身份由 ~/.flyclaw/SOUL.md 提供（用户可编辑），首次运行自动生成默认内容。
    环境信息自动探测，平台提示按 channel 注入。
    工具使用指导根据实际注册的工具动态生成。
    """
    workspace_dir = "."
    if config:
        agents_cfg = getattr(config, "agents", None)
        if agents_cfg:
            raw_ws = getattr(agents_cfg, "workspace", ".") or "."
            workspace_dir = str(Path(raw_ws).expanduser().resolve())

    has_custom_prompt = bool(extra_system_prompt.strip())
    soul_content = _load_soul_md()

    lines: list[str] = []

    lines.append(soul_content)
    lines.append("")

    if has_custom_prompt:
        lines.append(extra_system_prompt.strip())
        lines.append("")

    lines.extend(_build_environment_hints(workspace_dir))
    lines.extend(_build_platform_hints(channel))

    sandbox_enabled = True
    if config:
        exec_cfg = getattr(getattr(config, "tools", None), "exec", None)
        if exec_cfg:
            sandbox_enabled = getattr(exec_cfg, "sandbox_enabled", True)
    lines.extend(_build_sandbox_hints(sandbox_enabled, workspace_dir))

    lines.extend(_build_tooling_rules(tools))
    lines.extend(_build_tool_guidance(tools))
    lines.extend(_build_safety())
    lines.extend(_build_skills_section(skills_prompt))
    lines.extend(_build_workspace(workspace_dir))
    if context_files:
        lines.extend(_build_bootstrap_context(context_files))

    return "\n".join(lines)
