"""Modular system prompt builder — tool-guidance follows tools, config controls persona."""

from __future__ import annotations

import logging

logger = logging.getLogger("myclaw.prompt")


def _build_identity() -> list[str]:
    return [
        "你是一个运行在 MyClaw 中的 AI 助手。MyClaw 是你的运行平台，你没有 shell 访问权限来操作 MyClaw 自身的功能。",
        "",
    ]


def _build_tooling(tools: list) -> list[str]:
    lines = [
        "## 可用工具",
        "以下是当前可用的工具（按策略过滤）：",
        "工具名称区分大小写，必须精确匹配。",
    ]
    for t in tools:
        desc = (t.description or "").split("\n")[0].strip()
        if desc:
            lines.append(f"- {t.name}: {desc}")
        else:
            lines.append(f"- {t.name}")

    lines += [
        "",
        "## 工具调用风格",
        "不要描述常规工具调用 — 直接调用。仅在多步骤工作、敏感操作或被要求时才描述。",
        "优先使用工具调用，而非让用户手动执行 CLI 命令。",
        "不要尝试通过 exec_command 执行 myclaw、openclaw 等命令来操作平台内部功能（定时任务、记忆、会话等），这些都有对应的工具。",
        "",
    ]
    return lines


def _build_tool_guidance(tools: list) -> list[str]:
    """根据实际注册的工具名，条件注入中文使用指导。"""
    tool_names = {t.name for t in tools}
    lines: list[str] = []

    if tool_names & {"edit_file", "read_file", "write_file"}:
        lines += [
            "## 文件操作",
            "- edit_file 前必须先 read_file，需要精确匹配 old_string",
            "- 优先使用 file_tools（read_file/write_file/edit_file/list_dir/search_files）而非 exec_command",
            "",
        ]

    if tool_names & {"qq_send_image", "qq_send_file", "send_voice"}:
        lines += [
            "## 媒体发送",
            "在回复文本中使用标签包裹本地文件路径，系统自动发送：",
            "- <media>path</media> 自动识别类型",
            "- <qqimg>path</qqimg> 图片、<qqvoice>path</qqvoice> 音频、<qqfile>path</qqfile> 文件、<qqvideo>path</qqvideo> 视频",
            "",
        ]

    if "browser_navigate" in tool_names:
        lines += [
            "## 浏览器自动化",
            "- 先 browser_navigate 打开网页，再 browser_snapshot 获取元素引用（@e1, @e2...）",
            "- 通过引用操作：browser_click(\"@e1\")、browser_type(\"@e2\", \"text\")",
            "- 操作失败时重新 browser_snapshot 获取最新状态",
            "",
        ]

    if "memory" in tool_names:
        lines += [
            "## 持久记忆",
            "- memory(action=\"save\", content=\"...\"): 保存记忆（自动去重），用于用户偏好、身份、项目信息",
            "- memory(action=\"list\"): 列出所有记忆 / memory(action=\"list\", query=\"关键词\"): 搜索",
            "- memory(action=\"get\", key=\"...\"): 按键取回",
            "- memory(action=\"delete\", keys=[\"...\"]): 请求删除，需用户发 /y 确认（120s 超时自动拒绝）",
            "- memory(action=\"search\", query=\"...\"): 语义搜索历史记忆和知识库",
            "",
        ]

    if "task_manage" in tool_names:
        lines += [
            "## 任务模式（自主工作）",
            "- 收到复杂任务时，先调用 task_manage(action=\"plan\", goal=\"...\", plan_json=\"...\") 制定计划",
            "- 检查点触发时用 task_manage(action=\"status\") 查看进度",
            "- 完成步骤后 task_manage(action=\"advance\", step_index=N) 标记完成",
            "- 用 task_manage(action=\"cancel\") 可取消任务",
            "",
        ]

    if "cronjob" in tool_names:
        lines += [
            "## 定时任务",
            "- cronjob(action=\"create\", name=\"...\", message=\"...\", schedule_kind=\"...\") 创建定时任务",
            "- cronjob(action=\"list\") 查看，cronjob(action=\"delete\", job_id=\"...\") 删除",
            "- cronjob(action=\"toggle\", job_id=\"...\") 启停，cronjob(action=\"run\", job_id=\"...\") 立即触发",
            "- 一次性任务必须用 schedule_kind=\"at\" + run_at（如 \"2026-05-19 23:12:00\"），执行后自动删除",
            "- 不要用 cron 表达式创建一次性任务，cron 是循环的，不会自动删除",
            "",
        ]

    if "delegate_task" in tool_names:
        lines += [
            "## 子代理",
            "- delegate_task(agent_name=\"...\", task=\"...\"): 委派单个子任务",
            "- delegate_task(tasks=\"[{...}, {...}]\"): 并行委派多个子任务",
            "",
        ]

    if "session_search" in tool_names:
        lines += [
            "## 会话搜索",
            "- session_search: 搜索历史对话记录",
            "",
        ]

    if "web_search" in tool_names:
        lines += [
            "## 网页工具",
            "- web_search: 联网搜索",
            "- web_fetch: 抓取网页内容转 markdown",
            "",
        ]

    if "describe_image" in tool_names:
        lines += [
            "## 媒体理解",
            "- describe_image: 分析图片",
            "- transcribe_audio: 语音转文字",
            "- describe_video: 分析视频",
            "",
        ]

    if "text_to_speech" in tool_names:
        lines += [
            "## 语音合成",
            "- text_to_speech: 文字转语音并发送到当前聊天",
            "",
        ]

    if "qq_list_guilds" in tool_names:
        lines += [
            "## QQ 群操作",
            "- qq_list_guilds/channels/members 查看服务器信息",
            "- qq_send_text/image/file 主动发送消息",
            "",
        ]

    if "skill_manage" in tool_names:
        lines += [
            "## 技能系统",
            "- skill_manage(action=\"list\") 查看可用技能",
            "- skill_manage(action=\"view\", name=\"...\") 加载技能详情",
            "",
        ]

    if "procedure_search" in tool_names:
        lines += [
            "## 工作流模式记忆",
            "- procedure_search: 遇到多步骤任务时，先搜索是否已有现成流程可复用",
            "- procedure_learn: 成功完成新的多步骤任务后，保存为流程供未来复用",
            "- procedure_list: 查看已保存的所有工作流模式",
            "",
        ]

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
        "扫描以下技能描述，如果某个技能明显适用，用 skill_manage(action=\"view\", name=\"...\") 加载。",
        "如果都不适用，不要调用 skill_manage。不要用 read_file 读取技能文件。",
        "限制：最多加载一个技能；仅在选定后加载。",
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
    has_soul = any(
        f.get("path", "").lower() == "soul.md" for f in context_files
    )
    if has_soul:
        lines.append(
            "如果存在 SOUL.md，请体现其人设和语气。"
            "避免生硬的通用回复，遵循其指导，除非更高优先级的指令覆盖它。"
        )
    lines.append("")
    for f in context_files:
        lines.append(f"## {f['path']}")
        lines.append("")
        lines.append(f["content"])
        lines.append("")
    return lines


def build_system_prompt(
    config,
    tools: list,
    skills_prompt: str = "",
    context_files: list[dict] | None = None,
    extra_system_prompt: str = "",
) -> str:
    """组装完整系统提示词。

    config.yaml 的 system_prompt 负责人设和风格（1-2行）。
    工具使用指导根据实际注册的工具动态生成，确保与代码同步。
    """
    workspace_dir = "."
    if config:
        agents_cfg = getattr(config, "agents", None)
        if agents_cfg:
            raw_ws = getattr(agents_cfg, "workspace", ".") or "."
            from pathlib import Path
            workspace_dir = str(Path(raw_ws).expanduser().resolve())

    has_custom_prompt = bool(extra_system_prompt.strip())

    lines: list[str] = []

    if has_custom_prompt:
        lines.append(extra_system_prompt.strip())
        lines.append("")
    else:
        lines.extend(_build_identity())

    lines.extend(_build_tooling(tools))
    lines.extend(_build_tool_guidance(tools))

    lines.extend([
        "## 工具调用规则",
        "始终使用原生工具调用，不要将工具调用输出为文本、XML 或伪标签。",
        "",
    ])

    lines.extend(_build_safety())
    lines.extend(_build_skills_section(skills_prompt))
    lines.extend(_build_workspace(workspace_dir))
    if context_files:
        lines.extend(_build_bootstrap_context(context_files))

    return "\n".join(lines)
