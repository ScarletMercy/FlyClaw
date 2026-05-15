"""Beads memory integration for MyClaw - persistent key-value memory via bd remember."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
from typing import Optional

from src.agent.tooldef import ToolDef

logger = logging.getLogger("myclaw.beads_tools")

_BD_PATH: str = ""
_BEADS_WORKSPACE: str = ""

# Patterns for auto-extracting memory-worthy content from user input
# These are intentionally strict — false negatives are fine, false positives cause noise.
_MEMORY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("preference", re.compile(
        r"请?记住|别忘了|以后(?:要|不要|别)|"
        r"不要用|不要做|别用|别做|我的风格|"
        r"我(?:偏好|习惯)(?:是|:|：|用)",
        re.IGNORECASE,
    )),
    ("identity", re.compile(
        r"我(?:的)?(?:名字|姓|号|昵称|网名|ID|身份)(?:是|叫)|"
        r"叫我|我叫|我是(?![a-z\s])|"
        r"我的(?:生日|年龄|地址|家乡|学校|公司|职位|职业|专业)",
        re.IGNORECASE,
    )),
    ("contact", re.compile(
        r"(?:邮箱|email|邮件|电话|手机号?|微信号?|QQ|Telegram|discord|github)(?:是|:|：|=)|"
        r"(?:@|邮箱|email).*?(?:是|:|：)|"
        r"1[3-9]\d{9}",
        re.IGNORECASE,
    )),
    ("project", re.compile(
        r"(?:我的|我们的)(?:项目|仓库|代码库|产品|系统|服务|网站|app|应用)|"
        r"(?:用了|使用|技术栈|框架|部署在|跑在|运行在)|"
        r"(?:公司|团队|组织|部门)",
        re.IGNORECASE,
    )),
    ("service", re.compile(
        r"(?:API|api|接口|地址|URL|url|域名|服务器|端口|数据库|redis|mysql|postgres|mongo)"
        r"(?:是|:|：|=|地址)",
        re.IGNORECASE,
    )),
]


def set_beads_workspace(workspace: str) -> None:
    global _BEADS_WORKSPACE
    _BEADS_WORKSPACE = workspace
    logger.info("Beads workspace set to: %s", workspace)
    # Set BEADS_DIR so bd CLI can find the workspace
    beads_dir = os.path.join(workspace, ".beads")
    os.environ["BEADS_DIR"] = beads_dir
    logger.info("BEADS_DIR set to: %s", beads_dir)
    # Ensure bd is in PATH so exec_command can also find it
    bd_dir = os.path.dirname(_find_bd())
    if bd_dir and bd_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bd_dir + os.pathsep + os.environ.get("PATH", "")
        logger.info("Added %s to PATH", bd_dir)


def _find_bd() -> str:
    global _BD_PATH
    if _BD_PATH:
        return _BD_PATH
    candidates = [
        os.path.join(os.environ.get("ProgramFiles", ""), "beads", "bd.exe"),
        "D:/tools/bd.exe",
        "D:/tools/beads/bd.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            _BD_PATH = c
            return c
    found = shutil.which("bd") or shutil.which("bd.exe")
    if found:
        _BD_PATH = found
        return found
    return "bd"


async def _bd(args: list[str], input_text: str = "", timeout: float = 15.0) -> str:
    bd = _find_bd()
    cmd = [bd] + args
    workspace = _BEADS_WORKSPACE or os.environ.get("MYCLAW_WORKSPACE", "")
    cwd = workspace if workspace and os.path.isdir(workspace) else None

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=input_text.encode() if input_text else None),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        logger.error("bd %s timed out after %.0fs", " ".join(args), timeout)
        return f"[error] bd {' '.join(args)} timed out"
    out = stdout.decode(errors="replace").strip()
    err = stderr.decode(errors="replace").strip()
    if proc.returncode != 0:
        logger.error("bd %s failed (%d): %s", " ".join(args), proc.returncode, err)
        return f"[error] bd {' '.join(args)} failed: {err[:300]}"
    return out


# ---------------------------------------------------------------------------
# Auto-extract & LLM judge (used by main.py passive memory)
# ---------------------------------------------------------------------------

def auto_extract_memory(user_input: str, ai_response: str) -> Optional[tuple[str, str]]:
    """Regex-based quick check. Returns (content, category) or None."""
    if not user_input or len(user_input.strip()) < 4:
        return None
    for category, pattern in _MEMORY_PATTERNS:
        if pattern.search(user_input):
            content = f"[{category}] {user_input[:80].strip()}"
            return content, category
    return None


_JUDGE_PROMPT = """\
判断下面的对话是否包含用户主动提供的、明确的、值得永久记住的事实信息。

只记住以下情况（必须同时满足"明确"和"永久有用"两个条件）：
- 用户直接说了自己的偏好/习惯（如"我喜欢""我习惯""以后请"）
- 用户直接说了自己的身份/联系方式（如名字、邮箱、电话）
- 用户直接说了项目/工作相关的固定信息（如技术栈、服务器地址）

以下情况一律不记：
- 闲聊、寒暄、问答（"在吗""帮我XX""今天天气"）
- 一次性指令（"打开XX""截图""运行XX"）
- 通用知识、讨论、解释
- 情绪表达（"太好了""烦死了"）
- 模糊/不确定的信息
- 对话中没有任何用户主动透露的个人事实

对话:
用户: {user}
助手: {ai}

严格判断。如果有疑问，就不记。

只输出 JSON:
值得记: {{"remember": true, "content": "具体事实内容(一句话)", "category": "preference|identity|contact|project|fact"}}
不值得记: {{"remember": false}}"""


async def judge_memory_with_llm(
    user_input: str,
    ai_response: str,
    model_name: str,
    base_url: str,
    api_key: str,
) -> Optional[str]:
    """Use a small LLM to judge. Returns content string or None."""
    from src.agent.client import ChatClient

    model = ChatClient(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        temperature=0.0,
    )
    prompt = _JUDGE_PROMPT.format(
        user=user_input[:200],
        ai=ai_response[:200],
    )
    try:
        resp = await model.chat([
            {"role": "user", "content": prompt},
        ])
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        if not data.get("remember"):
            return None
        content = data.get("content", "").strip()
        category = data.get("category", "fact")
        if content:
            return f"[{category}] {content}"
        return None
    except Exception as e:
        logger.debug("Memory judge error: %s", e)
        return None


_SESSION_JUDGE_PROMPT = """\
分析以下完整会话，提取所有值得永久记住的事实信息。

只提取以下情况：
- 用户明确表达的偏好/习惯（如"我喜欢""我习惯""以后请"）
- 用户的身份/联系方式（名字、邮箱、电话等）
- 项目/工作的固定信息（技术栈、服务器地址、工作流程）

忽略：
- 闲聊、一次性指令、通用知识讨论
- 情绪表达、模糊信息

对话历史（{turn_count} 轮）:
{conversation_summary}

输出 JSON 数组，每个元素格式:
[{"content": "具体事实(一句话)", "category": "preference|identity|contact|project|fact"}]

如果没有值得记忆的，输出 []。
严格判断，宁缺毋滥。"""


async def extract_session_end_memories(
    messages: list[dict],
    model_name: str,
    base_url: str,
    api_key: str,
) -> int:
    """会话结束时批量提取记忆，比逐轮提取更高效。
    
    Args:
        messages: 完整的会话消息列表
        model_name: 用于判断的小模型名称
        base_url: 模型 API 地址
        api_key: 模型 API 密钥
    
    Returns:
        成功保存的记忆数量
    """
    from src.agent.client import ChatClient

    # 1. 先用正则快速匹配所有用户消息
    user_messages = [m["content"] for m in messages if m.get("role") == "user"]
    extracted_count = 0
    
    for msg in user_messages:
        result = auto_extract_memory(msg, "")
        if result:
            content, category = result
            await save_memory(content, "")
            extracted_count += 1
    
    # 如果正则已经提取到记忆，跳过 LLM 判断（避免重复）
    if extracted_count > 0:
        return extracted_count
    
    # 2. 如果没有正则匹配，使用 LLM 对整个会话进行批量判断
    if not model_name or not user_messages:
        return 0
    
    # 构建对话摘要（限制长度避免 token 溢出）
    conversation_parts = []
    total_chars = 0
    max_chars = 4000  # 限制摘要长度
    
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role in ("user", "assistant"):
            part = f"{role}: {content[:200]}"
            if total_chars + len(part) > max_chars:
                break
            conversation_parts.append(part)
            total_chars += len(part)
    
    conversation_summary = "\n".join(conversation_parts)
    turn_count = len([m for m in messages if m.get("role") == "user"])
    
    prompt = _SESSION_JUDGE_PROMPT.format(
        turn_count=turn_count,
        conversation_summary=conversation_summary,
    )
    
    model = ChatClient(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        temperature=0.0,
    )
    
    try:
        resp = await model.chat([
            {"role": "user", "content": prompt},
        ])
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        
        memories = json.loads(text)
        if isinstance(memories, list):
            for mem in memories:
                if isinstance(mem, dict) and mem.get("content"):
                    content = mem["content"].strip()
                    category = mem.get("category", "fact")
                    await save_memory(f"[{category}] {content}", "")
                    extracted_count += 1
    except Exception as e:
        logger.debug("Session-end memory extraction error: %s", e)
    
    return extracted_count


# ---------------------------------------------------------------------------
# Passive save helper (used by main.py)
# ---------------------------------------------------------------------------

async def save_memory(content: str, key: str = "") -> str:
    """Save a memory via bd remember. Auto-generates key from content if not provided."""
    args = ["remember", content, "--json"]
    if key:
        args += ["--key", key]
    return await _bd(args)


# ---------------------------------------------------------------------------
# Tools exposed to the LLM
# ---------------------------------------------------------------------------

async def bd_remember(content: str, key: str = "") -> str:
    """Save a persistent memory that survives across sessions.

    Use this to remember user preferences, facts, decisions, or any important info.
    If a memory with the same key exists, it will be updated.

    Args:
        content: The memory content to save.
        key: Optional explicit key. Auto-generated from content if empty.
    """
    return await save_memory(content, key)


async def bd_recall(key: str) -> str:
    """Retrieve a specific memory by its key.

    Args:
        key: The memory key to look up.
    """
    result = await _bd(["recall", key, "--json"])
    return result


async def bd_memories(query: str = "") -> str:
    """List all memories, or search by keyword.

    Args:
        query: Optional search keyword. Empty to list all.
    """
    args = ["memories", "--json"]
    if query:
        args.insert(1, query)
    return await _bd(args)


async def bd_forget(key: str) -> str:
    """Delete a memory by its key.

    Args:
        key: The memory key to delete.
    """
    return await _bd(["forget", key])


def get_tools() -> list[ToolDef]:
    return [
        ToolDef.from_function(bd_remember),
        ToolDef.from_function(bd_recall),
        ToolDef.from_function(bd_memories),
        ToolDef.from_function(bd_forget),
    ]
