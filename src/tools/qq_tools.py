"""QQ Bot tools for MyClaw - guild and channel management."""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Optional

logger = logging.getLogger("myclaw.qq_tools")

# Auto-injected by QQ channel before each message callback
_current_qq_chat_id: ContextVar[str] = ContextVar("_current_qq_chat_id", default="")


def set_current_qq_chat_id(chat_id: str):
    _current_qq_chat_id.set(chat_id)


def _get_qq_channel():
    from src.channels.qq import get_qq_channel

    return get_qq_channel()


def _get_http_and_token():
    """Get the QQ channel's http client and token manager."""
    ch = _get_qq_channel()
    if ch and ch._http_client and ch._token_manager:
        return ch._http_client, ch._token_manager
    return None, None


async def _qq_get(path: str, description: str = "QQ API"):
    """Authenticated GET to QQ API."""
    import httpx
    from src.channels.qq import API_BASE
    from src.channels.base import api_request_with_retry

    http_client, token_mgr = _get_http_and_token()
    if not http_client or not token_mgr:
        ch = _get_qq_channel()
        if ch is None:
            logger.error("%s: QQ channel not initialized (get_qq_channel() returned None)", description)
        elif ch._token_manager is None:
            logger.error("%s: QQ channel token_manager is None — start() was never called or failed before auth", description)
        elif ch._http_client is None:
            logger.error("%s: QQ channel http_client is None — start() failed after auth (before httpx client created)", description)
        return None

    token = await token_mgr.get_token()
    headers = {"Authorization": f"QQBot {token}"}

    try:
        resp = await api_request_with_retry(
            lambda: http_client.get(f"{API_BASE}{path}", headers=headers),
            description=description,
        )
        if resp.status_code == 401:
            token_mgr.clear_cache()
            token = await token_mgr.get_token()
            headers["Authorization"] = f"QQBot {token}"
            resp = await api_request_with_retry(
                lambda: http_client.get(f"{API_BASE}{path}", headers=headers),
                description=description,
            )
        if resp.status_code >= 400:
            logger.error("%s failed: %d %s", description, resp.status_code, resp.text[:200])
            return None
        return resp.json()
    except Exception as e:
        logger.error("%s error: %s", description, e)
        return None


async def qq_list_guilds() -> str:
    """List all guilds (servers) the QQ Bot has joined.

    Returns guild names and IDs.
    """
    data = await _qq_get("/users/@me/guilds", description="QQ list guilds")
    if data is None:
        return "[error] Failed to list guilds — check logs for details (channel not started, auth failed, or API error)"
    if not isinstance(data, list):
        return f"[error] Unexpected response: {data}"
    if not data:
        return "No guilds found."
    lines = []
    for g in data[:20]:
        lines.append(f"- {g.get('name', '?')} (id: {g.get('id', '?')})")
    result = "\n".join(lines)
    if len(data) > 20:
        result += f"\n... and {len(data) - 20} more"
    return result


async def qq_list_channels(guild_id: str) -> str:
    """List all channels in a QQ guild.

    Args:
        guild_id: The guild ID to list channels for.
    """
    data = await _qq_get(f"/guilds/{guild_id}/channels", description="QQ list channels")
    if not data:
        return "[error] Failed to list channels"
    if not isinstance(data, list):
        return f"[error] Unexpected response: {data}"
    if not data:
        return "No channels found."
    lines = []
    for ch in data[:20]:
        ch_type = ch.get("type", 0)
        type_names = {0: "text", 1: "voice", 2: "category", 3: "live"}
        type_name = type_names.get(ch_type, f"type={ch_type}")
        lines.append(f"- {ch.get('name', '?')} (id: {ch.get('id', '?')}, type: {type_name})")
    result = "\n".join(lines)
    if len(data) > 20:
        result += f"\n... and {len(data) - 20} more"
    return result


async def qq_list_members(guild_id: str, limit: int = 20) -> str:
    """List members in a QQ guild.

    Args:
        guild_id: The guild ID to list members for.
        limit: Max members to return (1-100, default 20).
    """
    limit = max(1, min(100, limit))
    data = await _qq_get(
        f"/guilds/{guild_id}/members?limit={limit}",
        description="QQ list members",
    )
    if not data:
        return "[error] Failed to list members"
    if not isinstance(data, list):
        return f"[error] Unexpected response: {data}"
    if not data:
        return "No members found."
    lines = []
    for m in data[:limit]:
        nick = m.get("nick", m.get("user", {}).get("username", "?"))
        uid = m.get("user", {}).get("id", "?")
        roles = ", ".join(m.get("roles", []))
        line = f"- {nick} (id: {uid})"
        if roles:
            line += f" [roles: {roles}]"
        lines.append(line)
    return "\n".join(lines)


async def qq_get_member(guild_id: str, user_id: str) -> str:
    """Get detailed info about a QQ guild member.

    Args:
        guild_id: The guild ID.
        user_id: The user ID to look up.
    """
    data = await _qq_get(
        f"/guilds/{guild_id}/members/{user_id}",
        description="QQ get member",
    )
    if not data:
        return "[error] Failed to get member info"
    user = data.get("user", {})
    lines = [
        f"Username: {user.get('username', '?')}",
        f"ID: {user.get('id', '?')}",
        f"Nickname: {data.get('nick', '')}",
        f"Roles: {', '.join(data.get('roles', []))}",
        f"Joined at: {data.get('joined_at', '?')}",
    ]
    return "\n".join(lines)


async def qq_send_text(chat_id: str = "", text: str = "", reply_to: Optional[str] = None) -> str:
    """Send a text message to a QQ user or group via the bot.

    Args:
        chat_id: Chat ID in format 'c2c:openid', 'group:openid', 'channel:id', or 'dm:id'. Leave empty to reply in current chat.
        text: Message text content.
        reply_to: Optional message ID to reply to.
    """
    chat_id = chat_id or _current_qq_chat_id.get("")
    if not chat_id:
        return "[error] No chat_id provided and no current QQ chat context"
    ch = _get_qq_channel()
    if not ch:
        return "[error] QQ channel not initialized"
    result = await ch.send_text(chat_id, text, reply_to)
    if result is not None:
        return "Message sent."
    return "[error] Failed to send message"


async def qq_send_image(chat_id: str = "", image_key: str = "") -> str:
    """Send an image to a QQ user or group.

    Args:
        chat_id: Chat ID in format 'c2c:openid', 'group:openid', etc. Leave empty to reply in current chat.
        image_key: Image URL or local file path.
    """
    chat_id = chat_id or _current_qq_chat_id.get("")
    if not chat_id:
        return "[error] No chat_id provided and no current QQ chat context"
    ch = _get_qq_channel()
    if not ch:
        return "[error] QQ channel not initialized"
    ok = await ch.send_image(chat_id, image_key)
    return "Image sent." if ok else "[error] Failed to send image"


async def qq_send_file(chat_id: str = "", file_key: str = "") -> str:
    """Send a file to a QQ user or group (C2C and group only).

    Args:
        chat_id: Chat ID in format 'c2c:openid' or 'group:openid'. Leave empty to reply in current chat.
        file_key: File URL or local file path.
    """
    chat_id = chat_id or _current_qq_chat_id.get("")
    if not chat_id:
        return "[error] No chat_id provided and no current QQ chat context"
    ch = _get_qq_channel()
    if not ch:
        return "[error] QQ channel not initialized"
    ok = await ch.send_file(chat_id, file_key)
    return "File sent." if ok else "[error] Failed to send file"


def get_tools() -> list:
    from src.agent.tooldef import ToolDef
    return [
        ToolDef.from_function(qq_list_guilds),
        ToolDef.from_function(qq_list_channels),
        ToolDef.from_function(qq_list_members),
        ToolDef.from_function(qq_get_member),
        ToolDef.from_function(qq_send_text),
        ToolDef.from_function(qq_send_image),
        ToolDef.from_function(qq_send_file),
    ]
