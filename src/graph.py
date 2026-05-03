from __future__ import annotations

import logging
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Annotated, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict

from src.skills.types import Skill

logger = logging.getLogger("myclaw.graph")

_COMPACT_THRESHOLD_TOKENS = 80000
_CHARS_PER_TOKEN = 4


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    system_prompt: str
    sender_id: str
    chat_id: str
    chat_type: str
    message_id: str


def create_agent_state(
    sender_id: str,
    chat_id: str,
    message_text: str,
    chat_type: str = "p2p",
    message_id: str = "",
    system_prompt: str = "",
) -> AgentState:
    """Factory function to create AgentState with validation.

    Args:
        sender_id: ID of the user sending the message
        chat_id: ID of the chat/conversation
        message_text: The message content
        chat_type: Type of chat ('p2p' or 'group')
        message_id: Optional message ID
        system_prompt: Optional system prompt override

    Returns:
        Validated AgentState dictionary
    """
    if not sender_id:
        raise ValueError("sender_id is required")
    if not chat_id:
        raise ValueError("chat_id is required")
    if not message_text:
        raise ValueError("message_text is required")
    if chat_type not in ("p2p", "group"):
        raise ValueError(f"chat_type must be 'p2p' or 'group', got '{chat_type}'")

    return {
        "messages": [HumanMessage(content=message_text)],
        "system_prompt": system_prompt,
        "sender_id": sender_id,
        "chat_id": chat_id,
        "chat_type": chat_type,
        "message_id": message_id or str(uuid.uuid4()),
    }


def _build_runtime_info() -> str:
    import zoneinfo
    try:
        tz = zoneinfo.ZoneInfo("Asia/Shanghai")
    except Exception:
        tz = timezone.utc
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M %Z")
    return (
        f"Runtime: Python {sys.version.split()[0]} on {platform.system()} {platform.release()}\n"
        f"Current time: {now}"
    )


def _estimate_tokens(messages: list[BaseMessage]) -> int:
    total = 0
    for m in messages:
        if isinstance(m.content, str):
            total += len(m.content) // _CHARS_PER_TOKEN
        elif isinstance(m.content, list):
            for part in m.content:
                if isinstance(part, dict) and "text" in part:
                    total += len(part["text"]) // _CHARS_PER_TOKEN
                elif isinstance(part, str):
                    total += len(part) // _CHARS_PER_TOKEN
                elif hasattr(part, "text"):
                    total += len(part.text) // _CHARS_PER_TOKEN
        total += 10
    return total


def _compact_messages(
    messages: list[BaseMessage],
    max_tokens: int = _COMPACT_THRESHOLD_TOKENS,
) -> list[BaseMessage]:
    if not messages:
        return messages
    estimated = _estimate_tokens(messages)
    if estimated <= max_tokens:
        return messages

    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]

    if not non_system:
        return messages

    tool_related = []
    regular = []
    for m in non_system:
        if isinstance(m, ToolMessage) or (isinstance(m, AIMessage) and m.tool_calls):
            tool_related.append(m)
        else:
            regular.append(m)

    keep_recent = max(6, len(non_system) // 3)
    kept = non_system[-keep_recent:]
    pruned = non_system[:-keep_recent]

    if not pruned:
        return messages

    summaries = []
    for m in pruned:
        if isinstance(m, ToolMessage):
            tool_name = getattr(m, "name", "unknown")
            content = m.content if isinstance(m.content, str) else str(m.content)
            summaries.append(f"[Tool({tool_name})]: {content[:150]}")
        elif isinstance(m, AIMessage) and m.tool_calls:
            calls = ", ".join(tc.get("name", "?") for tc in m.tool_calls)
            text = m.content if isinstance(m.content, str) else ""
            entry = f"Assistant: called [{calls}]"
            if text:
                entry += f" {text[:100]}"
            summaries.append(entry)
        elif isinstance(m, HumanMessage):
            text = m.content if isinstance(m.content, str) else str(m.content)[:200]
            summaries.append(f"User: {text[:200]}")
        elif isinstance(m, AIMessage) and m.content:
            text = m.content if isinstance(m.content, str) else str(m.content)[:200]
            summaries.append(f"Assistant: {text[:200]}")

    summary_text = "[Earlier conversation summarized]\n" + "\n".join(summaries[-20:])
    summary_msg = SystemMessage(content=summary_text)

    result = system_msgs + [summary_msg] + kept
    logger.info(
        "Compacted messages: %d → %d (estimated %d → %d tokens)",
        len(messages),
        len(result),
        estimated,
        _estimate_tokens(result),
    )
    return result


class FallbackModelChain:
    def __init__(self, primary: BaseChatModel, fallbacks: list[BaseChatModel] = None):
        self.primary = primary
        self.fallbacks = fallbacks or []
        self._all = [primary] + self.fallbacks
        self._cooldowns: dict[int, float] = {}  # keyed by id(model)

    async def ainvoke(self, messages, tools=None, **kwargs):
        now = time.time()
        errors = []
        for i, model in enumerate(self._all):
            cooldown_until = self._cooldowns.get(id(model), 0)
            if now < cooldown_until:
                continue
            try:
                m = model.bind_tools(tools) if tools else model
                result = await m.ainvoke(messages, **kwargs)
                return result
            except Exception as e:
                errors.append((i, e))
                error_str = str(e).lower()
                cooldown = 0
                if "rate" in error_str or "429" in error_str:
                    cooldown = 30
                elif "overload" in error_str or "503" in error_str or "529" in error_str:
                    cooldown = 60
                elif "auth" in error_str or "401" in error_str or "403" in error_str:
                    cooldown = 300
                elif "billing" in error_str or "quota" in error_str:
                    cooldown = 3600
                if cooldown > 0:
                    self._cooldowns[id(model)] = now + cooldown
                    logger.warning(
                        "Model %d (%s) failed, cooldown %ds: %s",
                        i,
                        type(model).__name__,
                        cooldown,
                        e,
                    )
                else:
                    logger.warning("Model %d failed: %s", i, e)
        if errors:
            raise errors[-1][1]
        raise RuntimeError("No models available")


def create_agent_graph(
    model_or_chain,
    tools: list[BaseTool],
    system_prompt: str,
    skills: list[Skill] | None = None,
    skills_budget: int = 30000,
    context_window_tokens: int = 100000,
    config=None,
) -> StateGraph:
    from src.skills.prompt import build_skills_prompt

    skills_prompt = ""
    if skills:
        skills_prompt = build_skills_prompt(skills, budget=skills_budget)

    # Pre-filter tools at compilation time if no policy is needed
    # If policy is enabled, we'll filter per-request based on sender_id
    _use_tool_policy = config is not None

    # Enforce max_tool_rounds: when reached, agent gets no tools so it must respond with text
    _max_tool_rounds = getattr(config, "agents", None)
    if _max_tool_rounds is None:
        _max_tool_rounds = 0
    else:
        _max_tool_rounds = getattr(_max_tool_rounds, "max_tool_rounds", 0) or 0

    all_tools = tools

    async def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        if not messages:
            return {"messages": []}

        compacted = _compact_messages(messages, max_tokens=int(context_window_tokens * 0.7))

        sp = state.get("system_prompt", system_prompt) or system_prompt
        runtime_info = _build_runtime_info()
        parts = [sp]
        if skills_prompt:
            parts.append(skills_prompt)
        parts.append(f"---\n{runtime_info}")
        full_system = "\n\n".join(parts)

        all_messages = [SystemMessage(content=full_system)] + list(compacted)

        active_tools = tools
        if _use_tool_policy:
            from src.tools.policy import apply_tool_policy

            active_tools = apply_tool_policy(tools, state.get("sender_id", ""), config)

        # Enforce max_tool_rounds: count ToolMessages only in current turn
        # (since the last HumanMessage), not across entire session history
        if _max_tool_rounds > 0:
            turn_messages = messages
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], HumanMessage):
                    turn_messages = messages[i:]
                    break
            tool_call_count = sum(1 for m in turn_messages if isinstance(m, ToolMessage))
            if tool_call_count >= _max_tool_rounds:
                logger.info("Max tool rounds (%d) reached in current turn, not binding tools", _max_tool_rounds)
                active_tools = []

        if active_tools:
            if isinstance(model_or_chain, BaseChatModel):
                response = await model_or_chain.bind_tools(active_tools).ainvoke(all_messages)
            else:
                response = await model_or_chain.ainvoke(all_messages, tools=active_tools)
        else:
            response = await model_or_chain.ainvoke(all_messages)
        return {"messages": [response]}

    async def approval_aware_tool_node(state: AgentState) -> dict:
        sender_id = state.get("sender_id", "")
        if _use_tool_policy:
            from src.tools.policy import apply_tool_policy

            filtered = apply_tool_policy(tools, sender_id, config)
        else:
            filtered = tools

        local_tool_node = ToolNode(filtered, handle_tool_errors=False)

        try:
            start_time = time.monotonic()
            result_state = await local_tool_node.ainvoke(state)
            duration = time.monotonic() - start_time

            # Audit log tool calls
            if _use_tool_policy and config.tools.exec.audit_log:
                try:
                    from src.hooks.command_logger import log_tool_call
                    last_msg = result_state["messages"][-1] if result_state.get("messages") else None
                    if isinstance(last_msg, ToolMessage):
                        tool_name = getattr(last_msg, "name", "unknown")
                        args = {}
                        ai_msg = None
                        for msg in reversed(state.get("messages", [])):
                            if isinstance(msg, AIMessage) and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    if tc.get("name") == tool_name:
                                        args = tc.get("args", {})
                                        break
                                if args:
                                    break
                        log_tool_call(
                            tool_name=tool_name,
                            args=args,
                            sender_id=sender_id,
                            chat_id=state.get("chat_id", ""),
                            success=True,
                            duration=duration,
                        )
                except Exception:
                    pass

            return result_state
        except Exception as e:
            from src.tools.exec import ApprovalNeededError
            from src.tools.exceptions import ToolExecutionError

            if isinstance(e, ApprovalNeededError):
                from src.tools.approval import get_approval_manager

                mgr = get_approval_manager()
                cmd = getattr(e, "command", "")
                denylisted = getattr(e, "denylisted", False)

                req = mgr.request_approval(
                    tool_name="exec_command",
                    args=cmd,
                    sender_id=sender_id,
                    chat_id=state.get("chat_id", ""),
                    message_id=state.get("message_id", ""),
                )

                decision = interrupt(
                    {
                        "type": "approval_request",
                        "request_id": req.id,
                        "tool_name": "exec_command",
                        "command_preview": cmd[:200],
                        "denylisted": denylisted,
                    }
                )

                if decision in ("allow_once", "allow_always"):
                    mgr._durable.setdefault("exec_command", [])
                    digest = mgr._make_digest("exec_command", cmd[:200])
                    if digest not in mgr._durable["exec_command"]:
                        mgr._durable["exec_command"].append(digest)
                        if decision == "allow_always":
                            mgr._save_durable()

                    try:
                        result_state = await local_tool_node.ainvoke(state)
                        return result_state
                    finally:
                        if decision == "allow_once":
                            if digest in mgr._durable.get("exec_command", []):
                                mgr._durable["exec_command"].remove(digest)
                else:
                    last_ai = None
                    for msg in reversed(state.get("messages", [])):
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            last_ai = msg
                            break
                    tool_call_id = (
                        last_ai.tool_calls[0]["id"]
                        if last_ai and last_ai.tool_calls
                        else "approval_denied"
                    )
                    denial_msg = ToolMessage(
                        content=f"[Command denied by user: {cmd[:100]}]",
                        tool_call_id=tool_call_id,
                    )
                    return {"messages": [denial_msg]}
            elif isinstance(e, ToolExecutionError):
                last_ai = None
                for msg in reversed(state.get("messages", [])):
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        last_ai = msg
                        break
                tool_call_id = (
                    last_ai.tool_calls[0]["id"] if last_ai and last_ai.tool_calls else "tool_error"
                )
                error_msg = ToolMessage(
                    content=f"[error] {str(e)}",
                    tool_call_id=tool_call_id,
                )
                return {"messages": [error_msg]}
            raise

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", approval_aware_tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    return graph


def _make_chat_model(
    provider: str,
    name: str,
    temperature: float,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BaseChatModel:
    """Create a chat model instance.

    Supports:
    - "anthropic" — Anthropic API (via langchain-anthropic)
    - "openai" — OpenAI API (via langchain-openai)
    - Any OpenAI-compatible provider (DeepSeek, Groq, Together, Ollama, vLLM, etc.)
      by setting provider="openai" with a custom base_url.
    """
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {"model": name, "temperature": temperature}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        return ChatAnthropic(**kwargs)
    else:
        # "openai" or any OpenAI-compatible provider
        from langchain_openai import ChatOpenAI

        kwargs = {"model": name, "temperature": temperature}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        return ChatOpenAI(**kwargs)


def create_model(config) -> BaseChatModel:
    return _make_chat_model(
        config.model.provider,
        config.model.name,
        config.model.temperature,
        base_url=config.model.base_url,
        api_key=config.model.api_key,
    )


def create_model_chain(config) -> FallbackModelChain:
    primary = _make_chat_model(
        config.model.provider,
        config.model.name,
        config.model.temperature,
        base_url=config.model.base_url,
        api_key=config.model.api_key,
    )
    fallbacks = []
    for fb in config.model.fallbacks:
        try:
            m = _make_chat_model(
                fb.provider,
                fb.name,
                config.model.temperature,
                base_url=getattr(fb, "base_url", None),
                api_key=getattr(fb, "api_key", None),
            )
            fallbacks.append(m)
            logger.info("Fallback model: %s/%s", fb.provider, fb.name)
        except Exception as e:
            logger.warning("Failed to init fallback %s/%s: %s", fb.provider, fb.name, e)

    return FallbackModelChain(primary, fallbacks)


def collect_tools(config) -> list[BaseTool]:
    from src.tools.registry import get_tool_registry

    registry = get_tool_registry()

    if not registry._registrations:
        _register_builtin_tools(config)

    return registry.collect()


def _register_builtin_tools(config) -> None:
    from src.tools.registry import get_tool_registry

    registry = get_tool_registry()

    @registry.register
    def _collect_exec_tools() -> list[BaseTool]:
        from src.tools.exec import exec_command

        if config.tools.exec.enabled:
            logger.info("Tool registered: exec_command")
            return [exec_command]
        return []

    @registry.register
    def _collect_web_search_tools() -> list[BaseTool]:
        from src.tools.web_search import web_search

        if config.tools.web_search.enabled:
            logger.info("Tool registered: web_search")
            return [web_search]
        return []

    @registry.register
    def _collect_web_fetch_tools() -> list[BaseTool]:
        from src.tools.web_fetch import web_fetch

        if config.tools.web_fetch.enabled:
            logger.info("Tool registered: web_fetch")
            return [web_fetch]
        return []

    @registry.register
    def _collect_feishu_tools() -> list[BaseTool]:
        from src.tools.feishu_tools import (
            feishu_send_message,
            feishu_send_card,
            feishu_create_document,
            feishu_write_document,
            feishu_get_doc_content,
            feishu_get_chat_info,
            feishu_get_user_info,
            feishu_get_chat_member_list,
            feishu_get_message_list,
            feishu_recall_message,
            feishu_create_chat,
            feishu_create_folder,
            feishu_drive_list,
            feishu_drive_upload,
            feishu_create_calendar_event,
            feishu_list_calendar_events,
            feishu_create_bitable,
            feishu_bitable_list_records,
            feishu_bitable_add_record,
            # Doc advanced ops
            feishu_doc_append,
            feishu_doc_insert,
            feishu_doc_list_blocks,
            feishu_doc_get_block,
            feishu_doc_update_block,
            feishu_doc_delete_block,
            feishu_doc_create_table,
            feishu_doc_insert_table_row,
            feishu_doc_insert_table_col,
            feishu_doc_delete_table_rows,
            feishu_doc_delete_table_cols,
            # Drive supplement
            feishu_drive_info,
            feishu_drive_move,
            feishu_drive_delete,
            feishu_drive_list_comments,
            feishu_drive_add_comment,
            feishu_drive_reply_comment,
            # Bitable supplement
            feishu_bitable_get_meta,
            feishu_bitable_list_fields,
            feishu_bitable_get_record,
            feishu_bitable_update_record,
            feishu_bitable_create_field,
            # Wiki
            feishu_wiki_list_spaces,
            feishu_wiki_list_nodes,
            feishu_wiki_get_node,
            feishu_wiki_create_node,
            feishu_wiki_move_node,
            feishu_wiki_rename_node,
            # Permissions
            feishu_perm_list_members,
            feishu_perm_add_member,
            feishu_perm_remove_member,
            # Chat supplement
            feishu_get_member_info,
            # Doc image upload
            feishu_doc_upload_image,
        )

        tools_list: list[BaseTool] = []
        if config.channels.feishu.enabled:
            tools_list.extend([
                feishu_send_message,
                feishu_send_card,
                feishu_create_document,
                feishu_write_document,
                feishu_get_doc_content,
                feishu_get_chat_info,
                feishu_get_user_info,
                feishu_get_chat_member_list,
                feishu_get_message_list,
                feishu_recall_message,
                feishu_create_chat,
                feishu_create_folder,
                feishu_drive_list,
                feishu_drive_upload,
                feishu_create_calendar_event,
                feishu_list_calendar_events,
                feishu_create_bitable,
                feishu_bitable_list_records,
                feishu_bitable_add_record,
                feishu_doc_append,
                feishu_doc_insert,
                feishu_doc_list_blocks,
                feishu_doc_get_block,
                feishu_doc_update_block,
                feishu_doc_delete_block,
                feishu_doc_create_table,
                feishu_doc_insert_table_row,
                feishu_doc_insert_table_col,
                feishu_doc_delete_table_rows,
                feishu_doc_delete_table_cols,
                feishu_drive_info,
                feishu_drive_move,
                feishu_drive_delete,
                feishu_drive_list_comments,
                feishu_drive_add_comment,
                feishu_drive_reply_comment,
                feishu_bitable_get_meta,
                feishu_bitable_list_fields,
                feishu_bitable_get_record,
                feishu_bitable_update_record,
                feishu_bitable_create_field,
                feishu_wiki_list_spaces,
                feishu_wiki_list_nodes,
                feishu_wiki_get_node,
                feishu_wiki_create_node,
                feishu_wiki_move_node,
                feishu_wiki_rename_node,
                feishu_perm_list_members,
                feishu_perm_add_member,
                feishu_perm_remove_member,
                feishu_get_member_info,
                feishu_doc_upload_image,
            ])
            logger.info("Tool registered: feishu tools (%d)", len(tools_list))

            from src.tools.media_tools import send_image_to_chat, send_file_to_chat

            tools_list.append(send_image_to_chat)
            tools_list.append(send_file_to_chat)
            logger.info("Tool registered: send_image_to_chat, send_file_to_chat")
        return tools_list

    @registry.register
    def _collect_qq_tools() -> list[BaseTool]:
        from src.tools.qq_tools import (
            qq_list_guilds,
            qq_list_channels,
            qq_list_members,
            qq_get_member,
            qq_send_text,
            qq_send_image,
            qq_send_file,
        )

        tools_list: list[BaseTool] = []
        if config.channels.qq.enabled:
            tools_list.extend([
                qq_list_guilds,
                qq_list_channels,
                qq_list_members,
                qq_get_member,
                qq_send_text,
                qq_send_image,
                qq_send_file,
            ])
            logger.info("Tool registered: QQ tools (%d)", len(tools_list))
        return tools_list

    @registry.register
    def _collect_file_tools() -> list[BaseTool]:
        from src.tools.file_tools import read_file, write_file, edit_file, list_dir, grep_files, glob_files

        logger.info("Tool registered: read_file, write_file, edit_file, list_dir, grep_files, glob_files")
        return [read_file, write_file, edit_file, list_dir, grep_files, glob_files]

    @registry.register
    def _collect_cron_tools() -> list[BaseTool]:
        if config.cron.enabled:
            from src.tools.cron_tools import cron_list, cron_add, cron_delete, cron_toggle, cron_run

            logger.info("Tool registered: cron_list, cron_add, cron_delete, cron_toggle, cron_run")
            return [cron_list, cron_add, cron_delete, cron_toggle, cron_run]
        return []

    @registry.register
    def _collect_media_understanding_tools() -> list[BaseTool]:
        if config.tools.media_understanding.enabled:
            from src.tools.media_understanding_tools import describe_image, transcribe_audio, describe_video

            logger.info("Tool registered: describe_image, transcribe_audio, describe_video")
            return [describe_image, transcribe_audio, describe_video]
        return []

    @registry.register
    def _collect_subagent_tools() -> list[BaseTool]:
        if getattr(config.agents, "subagents", None) and config.agents.subagents:
            from src.agents.delegate import delegate_task
            from src.tools.subagent_tools import subagent_status

            logger.info("Tool registered: delegate_task, subagent_status")
            return [delegate_task, subagent_status]
        return []

    @registry.register
    def _collect_memory_tools() -> list[BaseTool]:
        if getattr(config, "memory", None) and config.memory.enabled:
            from src.tools.memory_tools import memory_search

            logger.info("Tool registered: memory_search")
            return [memory_search]
        return []

    @registry.register
    def _collect_plugin_tools() -> list[BaseTool]:
        if getattr(config, "plugins", None) and config.plugins.enabled:
            try:
                from src.plugins.registry import get_plugin_registry as get_plugin_reg

                plugin_registry = get_plugin_reg()
                plugin_tools = plugin_registry.get_all_tools()
                if plugin_tools:
                    logger.info("Plugin tools registered: %d", len(plugin_tools))
                return plugin_tools
            except Exception as e:
                logger.warning("Failed to load plugin tools: %s", e)
        return []
