"""Background self-improvement review for skill learning.

After accumulating enough tool-calling iterations, spawns a background
agent that reviews the conversation and creates/updates skills.
Follows the Hermes-Agent pattern: nudge counter → background review fork.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger("flyclaw.skills.review")

SKILL_REVIEW_PROMPT = (
    "审查以上对话，更新技能库。保持积极——大多数会话至少产生一次技能更新，"
    "哪怕很小。什么都不做的审查是错失的学习机会。\n\n"
    "目标形态：类级(CLASS-LEVEL)技能，每个有丰富的 SKILL.md "
    "和 references/ 目录存放会话特定细节。"
    "不是一长串窄的每会话一条的微技能条目。\n\n"
    "值得捕获的信号（满足任一条即应行动）：\n"
    "  • 用户纠正了你的风格、语气、格式、冗长度。如'别再这样做了'"
    "'太啰嗦了''格式不对'等都是一等技能信号，不只是记忆信号。"
    "更新相关技能以嵌入偏好，使下一个会话一开始就知道。\n"
    "  • 用户纠正了你的工作流、方法或步骤序列。"
    "将纠正编码为管理该类任务的技能中的陷阱或显式步骤。\n"
    "  • 出现了非平凡的技巧、修复、变通方案、调试路径或工具使用模式，"
    "未来会话会受益的。捕获它。\n"
    "  • 本次会话加载或参考的技能过时、缺少步骤或有误。立即修补。\n\n"
    "优先级顺序——优先选择最早适用的动作，但一定要选一个：\n"
    "  1. 更新已加载的技能。回看对话中通过 /skill-name 加载或 "
    "skill_view(name=\"...\") 查看的技能。如果其中某个覆盖了"
    "新学习的领域，优先修补它。\n"
    "  2. 更新现有伞形技能（通过 skills_list() + "
    "skill_view(name=\"...\")）。如果没有已加载的技能适合，"
    "但存在一个类级技能匹配，修补它。添加子节、陷阱或扩展触发条件。\n"
    "  3. 在现有伞形下添加支撑文件。技能可以包含三种支撑文件"
    "——按类型使用正确目录：\n"
    "     • references/<topic>.md — 会话特定细节（错误转录、复现配方、"
    "Provider 怪癖）和浓缩知识库。\n"
    "     • templates/<name>.<ext> — 用于复制修改的起始文件。\n"
    "     • scripts/<name>.<ext> — 可静态重复运行的动作。\n"
    "通过 skill_manage(action=\"write_file\") 添加，"
    "file_path 以 references/、templates/ 或 scripts/ 开头。\n"
    "  4. 创建新的类级伞形技能。当没有现有技能覆盖该类别时使用。"
    "名称必须是类级的。名称不能是特定 PR 号、错误字符串、功能代号或 "
    "'fix-X / debug-Y / audit-Z-today' 会话产物。"
    "如果提议的名称只在今天有意义，就退回到 1、2 或 3。\n\n"
    "用户偏好嵌入（重要）：当用户表达风格/格式/工作流偏好时，"
    "更新属于 SKILL.md 正文，不只是记忆。记忆捕获'用户是谁'；"
    "技能捕获'如何为该用户完成此类任务'。"
    "当用户抱怨你处理任务的方式时，管理该任务的技能需要承载这个教训。\n\n"
    "如果发现两个现有技能重叠，在回复中注明——后台策展人负责大规模整合。\n\n"
    "不要捕获以下内容（会变成持久的自我约束，在环境变化时反过来伤害你）：\n"
    "  • 环境依赖的故障：缺失二进制、全新安装错误、迁移后路径不匹配、"
    "'command not found'、未配置凭证、未安装包。\n"
    "  • 对工具或功能的负面断言（'浏览器工具不能用'、'X 工具坏了'）。"
    "这些会固化为拒绝行为，在实际问题修复后数月仍被引用。\n"
    "  • 在会话结束前已解决的瞬时错误。如果重试成功，教训是重试模式，"
    "不是原始失败。\n"
    "  • 一次性任务叙述。用户要求'总结今天的市场'或'分析这个 PR'"
    "不构成需要技能的工作类别。\n\n"
    "如果工具因设置状态失败，捕获修复方法（安装命令、配置步骤、"
    "环境变量设置）到已有的设置或故障排除技能下——"
    "绝不要将'这个工具不能用'作为独立约束。\n\n"
    "'没有可保存的内容。'是一个真实选项，但不应是默认选项。"
    "如果会话顺利进行没有纠正也没有产生新技术，说'没有可保存的内容。'并停止。"
    "否则，行动。"
)

MEMORY_REVIEW_PROMPT = (
    "审查以上对话，考虑是否需要保存到记忆。\n\n"
    "关注：\n"
    "1. 用户是否透露了关于自己的信息——个人画像、愿望、偏好或个人细节？\n"
    "2. 用户是否表达了对你的行为方式的期望、工作风格或运作方式的期望？\n\n"
    "如果有值得保存的内容，用 memory 工具保存。"
    "如果没有值得保存的内容，说'没有可保存的内容。'并停止。"
)

COMBINED_REVIEW_PROMPT = (
    "审查以上对话，更新两件事：\n\n"
    "**记忆**：用户是谁。用户是否透露了个人画像、偏好、工作风格？"
    "如果有值得记住的，用 memory 工具保存。\n\n"
    "**技能**：用 skill_manage 更新技能库。"
    "目标：类级技能，不是窄的每会话微技能。\n"
    "优先级：1) 修补已加载/使用的技能 → 2) 修补现有伞形 → "
    "3) 添加支撑文件 → 4) 创建新伞形。\n"
    "用户纠正 = 一等技能信号。环境故障 = 不要捕获。\n\n"
    "如果都不需要更新，说'没有可保存的内容。'并停止。"
)


async def spawn_background_review(
    client: Any,
    tools: list,
    config: Any,
    messages_snapshot: list[dict],
    review_skills: bool = True,
    review_memory: bool = False,
    max_rounds: int = 16,
) -> str:
    """Spawn a background skill-review agent loop.

    Creates a lightweight AgentLoop with restricted tools (skills_list, skill_view, skill_manage + memory),
    inherits the parent's LLM client and config, and runs a self-improvement
    review over the conversation snapshot.

    Returns a compact summary of actions taken (e.g. "created skill X · patched skill Y").
    """
    from src.skills.provenance import (
        set_current_write_origin,
        reset_current_write_origin,
        BACKGROUND_REVIEW,
    )

    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        return await _run_review_loop(
            client=client,
            tools=tools,
            config=config,
            messages_snapshot=messages_snapshot,
            review_skills=review_skills,
            review_memory=review_memory,
            max_rounds=max_rounds,
        )
    except Exception as e:
        logger.warning("Background skill review failed: %s", e)
        return ""
    finally:
        reset_current_write_origin(token)


async def _run_review_loop(
    client: Any,
    tools: list,
    config: Any,
    messages_snapshot: list[dict],
    review_skills: bool,
    review_memory: bool,
    max_rounds: int,
) -> str:
    from src.agent.loop import AgentLoop
    from src.agent.state import AgentState, MemoryStateStore

    allowed = {"skills_list", "skill_view", "skill_manage", "memory"}
    review_tools = [t for t in tools if t.name in allowed]
    if not review_tools:
        logger.debug("No review-capable tools available, skipping background review")
        return ""

    if review_memory and review_skills:
        prompt = COMBINED_REVIEW_PROMPT
    elif review_memory:
        prompt = MEMORY_REVIEW_PROMPT
    else:
        prompt = SKILL_REVIEW_PROMPT

    state_store = MemoryStateStore()
    agent_loop = AgentLoop(
        client=client,
        tools=review_tools,
        state_store=state_store,
        config=config,
        skills_prompt="",
    )
    agent_loop._skill_nudge_interval = 0

    state = AgentState(
        messages=list(messages_snapshot) + [{"role": "user", "content": prompt}],
    )

    try:
        result_state = await agent_loop.run(
            state, "background:skill-review", max_rounds=max_rounds,
        )
    except Exception as e:
        logger.warning("Background review loop failed: %s", e)
        return ""

    actions = _summarize_actions(result_state.messages, len(messages_snapshot))

    if actions:
        summary = " · ".join(dict.fromkeys(actions))
        logger.info("Self-improvement review: %s", summary)
        return summary
    return ""


def _summarize_actions(messages: list[dict], pre_count: int) -> list[str]:
    """Scan review agent messages for successful tool actions and return summaries."""
    actions: list[str] = []
    for msg in messages[pre_count:]:
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        if '"action": "created"' in content and '"success": true' in content:
            _extract_name(actions, content, "created skill")
        elif '"action": "patched"' in content and '"success": true' in content:
            _extract_name(actions, content, "patched skill")
        elif '"action": "edited"' in content and '"success": true' in content:
            _extract_name(actions, content, "edited skill")
        elif '"action": "wrote_file"' in content and '"success": true' in content:
            _extract_name(actions, content, "wrote file in")
        elif '"action": "saved"' in content and '"success": true' in content:
            actions.append("saved memory")
    return actions


def _extract_name(actions: list[str], content: str, prefix: str) -> None:
    import json
    try:
        data = json.loads(content)
        name = data.get("skill", "")
        if isinstance(name, dict):
            name = name.get("name", "")
        if name:
            actions.append(f"{prefix} {name}")
    except Exception:
        pass
