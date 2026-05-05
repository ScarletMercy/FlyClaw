from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger("myclaw.feishu_tools")

_LARK_RATE_LIMIT_CODES = {99991400, 99991401, 99991402}
_LARK_MAX_RETRIES = 3


async def _lark_call_with_retry(call_fn, description: str = "Feishu API"):
    """Execute a lark SDK call with rate-limit retry logic."""
    for attempt in range(_LARK_MAX_RETRIES + 1):
        resp = await asyncio.to_thread(call_fn)
        code = getattr(resp, "code", None)
        if code not in _LARK_RATE_LIMIT_CODES:
            return resp
        if attempt >= _LARK_MAX_RETRIES:
            logger.warning("%s: rate limited (code=%s) after %d retries", description, code, _LARK_MAX_RETRIES)
            return resp
        wait = min(1.0 * (2**attempt), 8.0) + random.uniform(0, 0.5)
        logger.warning(
            "%s: rate limited (code=%s), retry %d/%d in %.1fs", description, code, attempt + 1, _LARK_MAX_RETRIES, wait
        )
        await asyncio.sleep(wait)
    return resp


async def _lark_thread(fn, *args, description: str = "Feishu API"):
    """Drop-in replacement for asyncio.to_thread with lark SDK rate-limit retry."""
    return await _lark_call_with_retry(lambda: fn(*args), description=description)


def _get_feishu_client():
    from src.channels.feishu import get_feishu_client

    return get_feishu_client()


def _resolve_api(api_path: str, client):
    """Resolve dotted API path like 'im.v1.chat.get' to a bound method."""
    obj = client
    for attr in api_path.split("."):
        obj = getattr(obj, attr)
    return obj


async def _feishu_call(api_path, req, *, extract=None, ok_msg=None):
    """Unified Feishu SDK call: client validation + retry + error handling + JSON formatting.

    Args:
        api_path: Dotted path to SDK method (e.g. "im.v1.chat.get") or callable(client).
        req: Request object to pass to the SDK method.
        extract: Callable(resp) -> result dict/list/str/None. If None, returns ok_msg.
        ok_msg: String to return on success when extract returns None.
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        fn = _resolve_api(api_path, client) if isinstance(api_path, str) else api_path(client)
        resp = await _lark_thread(fn, req)
        if not resp.success():
            return f"[error] {resp.code}: {resp.msg}"
        if extract:
            result = extract(resp)
            if result is None:
                return ok_msg or "No data found"
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        return ok_msg or "OK"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


@tool
async def feishu_send_message(chat_id: str, text: str, msg_type: str = "text") -> str:
    """Send a text or interactive message to a Feishu chat.

    Args:
        chat_id: The Feishu chat ID (e.g. "oc_xxx").
        text: Message text content. For text type, plain text. For interactive, JSON card content.
        msg_type: "text" (default) or "interactive" (card JSON).
    """
    from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

    content = text if msg_type == "interactive" else json.dumps({"text": text}, ensure_ascii=False)
    body = CreateMessageRequestBody.builder().receive_id(chat_id).msg_type(msg_type).content(content).build()
    req = CreateMessageRequest.builder().receive_id_type("chat_id").request_body(body).build()
    return await _feishu_call(
        "im.v1.message.create", req,
        extract=lambda r: f"Message sent to {chat_id}, message_id={r.data.message_id}" if r.data else None,
    )


@tool
async def feishu_get_user_info(user_id: str) -> str:
    """Get information about a Feishu user by open_id or user_id.

    Args:
        user_id: User's open_id, user_id, or union_id.
    """
    from lark_oapi.api.contact.v3 import GetUserRequest

    req = GetUserRequest.builder().user_id(user_id).user_id_type("open_id").build()
    return await _feishu_call(
        "contact.v3.user.get", req,
        extract=lambda r: {
            "name": r.data.user.name, "open_id": r.data.user.open_id,
            "union_id": r.data.user.union_id, "user_id": r.data.user.user_id,
            "email": r.data.user.email, "mobile": r.data.user.mobile,
            "department_ids": r.data.user.department_ids,
        } if r.data and r.data.user else None,
    )


@tool
async def feishu_get_chat_member_list(chat_id: str, page_size: int = 50) -> str:
    """Get the member list of a Feishu chat group.

    Args:
        chat_id: The chat ID to query.
        page_size: Number of members per page (max 50).
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        from lark_oapi.api.im.v1 import GetChatMembersRequest

        members = []
        page_token = None
        for _ in range(20):  # max 20 pages = 1000 members
            builder = GetChatMembersRequest.builder().chat_id(chat_id).page_size(page_size)
            if page_token:
                builder = builder.page_token(page_token)
            req = builder.build()
            resp = await _lark_thread(client.im.v1.chat_member.get, req)
            if not resp.success():
                return f"[error] {resp.code}: {resp.msg}"
            if resp.data and resp.data.items:
                for item in resp.data.items:
                    members.append(
                        {
                            "member_id": item.member_id,
                            "member_id_type": item.member_id_type,
                            "name": getattr(item, "name", ""),
                        }
                    )
            if not resp.data or not resp.data.has_more:
                break
            page_token = resp.data.page_token
        return json.dumps(members, ensure_ascii=False, default=str)
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


@tool
async def feishu_create_chat(name: str, description: str = "", user_ids: str = "") -> str:
    """Create a Feishu group chat.

    Args:
        name: Chat name.
        description: Chat description.
        user_ids: Comma-separated open_id list of members to add.
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        from lark_oapi.api.im.v1 import CreateChatRequest, CreateChatRequestBody

        body = CreateChatRequestBody.builder().name(name).description(description).build()
        req = CreateChatRequest.builder().request_body(body).build()
        resp = await _lark_thread(client.im.v1.chat.create, req)
        if not resp.success():
            return f"[error] {resp.code}: {resp.msg}"
        chat_id = resp.data.chat_id if resp.data else ""
        result = f"Chat created: {chat_id} (name={name})"

        if user_ids and chat_id:
            uid_list = [uid.strip() for uid in user_ids.split(",") if uid.strip()]
            for uid in uid_list[:50]:
                from lark_oapi.api.im.v1 import CreateChatMembersRequest, CreateChatMembersRequestBody

                member_body = CreateChatMembersRequestBody.builder().id_list([uid]).build()
                member_req = CreateChatMembersRequest.builder().chat_id(chat_id).request_body(member_body).build()
                mr = await _lark_thread(client.im.v1.chat_member.create, member_req)
                if not mr.success():
                    result += f"\n  Warning: failed to add {uid}: {mr.msg}"
        return result
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


@tool
async def feishu_get_chat_info(chat_id: str) -> str:
    """Get information about a Feishu chat group.

    Args:
        chat_id: The chat ID to query.
    """
    from lark_oapi.api.im.v1 import GetChatRequest

    req = GetChatRequest.builder().chat_id(chat_id).build()
    return await _feishu_call(
        "im.v1.chat.get", req,
        extract=lambda r: {
            "name": r.data.name, "chat_id": r.data.chat_id,
            "owner_id": r.data.owner_id, "member_count": r.data.user_count,
            "description": getattr(r.data, "description", ""),
            "chat_type": getattr(r.data, "chat_type", ""),
        } if r.data else None,
    )


@tool
async def feishu_get_doc_content(doc_token: str) -> str:
    """Read the content of a Feishu document (docx or wiki).

    Args:
        doc_token: The document token (the part after the last '/' in a Feishu doc URL).
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        from lark_oapi.api.docx.v1 import RawContentDocumentRequest

        def _read_doc(token: str) -> str:
            req = RawContentDocumentRequest.builder().document_id(token).build()
            resp = client.docx.v1.document.raw_content(req)
            if resp.success():
                return resp.data.content or "(empty document)"
            return ""

        content = await asyncio.to_thread(_read_doc, doc_token)
        if content:
            return content

        # Try as wiki page
        try:
            from lark_oapi.api.wiki.v2 import GetNodeSpaceRequest

            wiki_req = GetNodeSpaceRequest.builder().token(doc_token).build()
            wiki_resp = await _lark_thread(client.wiki.v2.space.get_node, wiki_req)
            if wiki_resp.success() and wiki_resp.data and wiki_resp.data.node:
                obj_token = wiki_resp.data.node.obj_token
                obj_type = wiki_resp.data.node.obj_type
                if obj_type == "docx" and obj_token:
                    content = await asyncio.to_thread(_read_doc, obj_token)
                    if content:
                        return content
                return f"[error] wiki node found but type={obj_type}, cannot read content"
        except Exception:
            pass
        return f"[error] Failed to read document: {doc_token}"
    except Exception as e:
        logger.error("feishu_get_doc_content error: %s", e, exc_info=True)
        return f"[error] {type(e).__name__}: {e}"


@tool
async def feishu_list_calendar_events(start_time: str = "", end_time: str = "", user_id: str = "") -> str:
    """List Feishu calendar events in a time range.

    Args:
        start_time: Start time in ISO format (e.g. "2026-04-18T00:00:00+08:00"). Defaults to 7 days ago.
        end_time: End time in ISO format. Defaults to 7 days from now.
        user_id: User's open_id (empty for bot's calendar).
    """
    from datetime import datetime as _dt, timedelta

    from lark_oapi.api.calendar.v4 import ListCalendarEventRequest

    now = _dt.now()
    start_ts = int(_dt.fromisoformat(start_time).timestamp() * 1000) if start_time else int((now - timedelta(days=7)).timestamp() * 1000)
    end_ts = int(_dt.fromisoformat(end_time).timestamp() * 1000) if end_time else int((now + timedelta(days=7)).timestamp() * 1000)
    cal_id = f"user_{user_id}" if user_id else ""
    req = ListCalendarEventRequest.builder().calendar_id(cal_id).start_time(start_ts).end_time(end_ts).page_size(50).build()
    return await _feishu_call(
        "calendar.v4.calendar_event.list", req,
        extract=lambda r: [{
            "summary": getattr(ev, "summary", ""),
            "start_time": str(getattr(ev, "start_time", "")),
            "end_time": str(getattr(ev, "end_time", "")),
            "description": getattr(ev, "description", "")[:200],
        } for ev in r.data.items] if r.data and r.data.items else None,
        ok_msg="No calendar events found",
    )


@tool
async def feishu_create_document(title: str, folder_token: str = "") -> str:
    """Create a new Feishu cloud document (docx) and return its URL.

    Args:
        title: Document title.
        folder_token: Target folder token (empty for root folder).
    """
    from lark_oapi.api.docx.v1 import CreateDocumentRequest, CreateDocumentRequestBody

    body = CreateDocumentRequestBody.builder().title(title).folder_token(folder_token or "").build()
    req = CreateDocumentRequest.builder().request_body(body).build()
    return await _feishu_call(
        "docx.v1.document.create", req,
        extract=lambda r: (
            f"Document created: {title}\n"
            + (f"URL: {r.data.document.url}\n" if getattr(r.data.document, "url", "") else "")
            + (f"Token: {r.data.document.document_id}" if getattr(r.data.document, "document_id", "") else "")
        ).rstrip("\n") if r.data and r.data.document else None,
    )


@tool
async def feishu_drive_upload(file_path: str, folder_token: str = "") -> str:
    """Upload a local file to Feishu Drive.

    Args:
        file_path: Local file path to upload.
        folder_token: Target folder token (empty for root folder).
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    import hashlib
    import os

    abs_path = os.path.abspath(file_path)
    if not os.path.isfile(abs_path):
        return f"[error] File not found: {abs_path}"
    try:
        from src.channels.feishu import _resolve_api_base, _get_tenant_token
        from src.config import load_config

        cfg = load_config()
        domain = cfg.channels.feishu.domain
        api_token = await _get_tenant_token(cfg.channels.feishu.app_id, cfg.channels.feishu.app_secret, domain)
        if not api_token:
            return "[error] Failed to get tenant token"
        api_base = _resolve_api_base(domain)

        file_name = os.path.basename(abs_path)
        file_size = os.path.getsize(abs_path)

        # Use httpx with proper multipart upload (bypasses SDK issues)
        # API: POST /drive/v1/files/upload_all
        # parent_type: "explorer" (root drive) or "wiki" (wiki space)
        import httpx

        form_fields = {
            "file_name": file_name,
            "size": str(file_size),
            "parent_type": "explorer",
        }
        if folder_token:
            form_fields["parent_node"] = folder_token

        with open(abs_path, "rb") as f:
            async with httpx.AsyncClient(timeout=60) as hc:
                resp = await hc.post(
                    f"{api_base}/drive/v1/files/upload_all",
                    headers={"Authorization": f"Bearer {api_token}"},
                    data=form_fields,
                    files={"file": (file_name, f)},
                )
        data = resp.json()
        if data.get("code") == 0:
            file_token = data.get("data", {}).get("file_token", "")
            return f"File uploaded: {file_name} (token: {file_token})"
        return f"[error] Upload failed: {data.get('code')}: {data.get('msg', '')}"
    except Exception as e:
        logger.error("feishu_drive_upload error: %s", e, exc_info=True)
        return f"[error] {type(e).__name__}: {e}"


@tool
async def feishu_write_document(doc_token: str, content: str) -> str:
    """Write content to an existing Feishu document (replace all content).

    Args:
        doc_token: Document token (from feishu_create_document or doc URL).
        content: Markdown content to write into the document.
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        from lark_oapi.api.docx.v1 import (
            ConvertDocumentRequest,
            ConvertDocumentRequestBody,
            GetDocumentBlockChildrenRequest,
            BatchDeleteDocumentBlockChildrenRequest,
            BatchDeleteDocumentBlockChildrenRequestBody,
            CreateDocumentBlockDescendantRequest,
            CreateDocumentBlockDescendantRequestBody,
        )

        if not content.strip():
            return "[error] No content to write"

        # 1. Get root block's existing children count (root block_id == document_id)
        get_req = (
            GetDocumentBlockChildrenRequest.builder().document_id(doc_token).block_id(doc_token).page_size(500).build()
        )
        get_resp = await asyncio.to_thread(client.docx.v1.document_block_children.get, get_req)
        children_count = 0
        if get_resp.success() and get_resp.data and get_resp.data.items:
            children_count = len(get_resp.data.items)

        # 2. Clear existing content (batch delete all children of root block)
        deleted = 0
        if children_count > 0:
            del_body = (
                BatchDeleteDocumentBlockChildrenRequestBody.builder()
                .start_index(0)
                .end_index(children_count - 1)
                .build()
            )
            del_req = (
                BatchDeleteDocumentBlockChildrenRequest.builder()
                .document_id(doc_token)
                .block_id(doc_token)
                .request_body(del_body)
                .build()
            )
            del_resp = await asyncio.to_thread(client.docx.v1.document_block_children.batch_delete, del_req)
            if del_resp.success():
                deleted = children_count
            else:
                logger.warning("Failed to clear document content: %s %s", del_resp.code, del_resp.msg)

        # 3. Convert markdown to Feishu blocks via SDK
        convert_body = ConvertDocumentRequestBody.builder().content_type("markdown").content(content).build()
        convert_req = ConvertDocumentRequest.builder().request_body(convert_body).build()
        convert_resp = await asyncio.to_thread(client.docx.v1.document.convert, convert_req)
        if not convert_resp.success() or not convert_resp.data:
            return f"[error] Markdown conversion failed: {convert_resp.code} {convert_resp.msg}"

        blocks = convert_resp.data.blocks or []
        first_level_ids = convert_resp.data.first_level_block_ids or []

        if not blocks or not first_level_ids:
            return f"Document cleared ({deleted} blocks), but conversion produced no blocks"

        # 4. Insert blocks via documentBlockDescendant.create
        insert_body = (
            CreateDocumentBlockDescendantRequestBody.builder()
            .children_id(first_level_ids)
            .index(-1)
            .descendants(blocks)
            .build()
        )
        insert_req = (
            CreateDocumentBlockDescendantRequest.builder()
            .document_id(doc_token)
            .block_id(doc_token)
            .request_body(insert_body)
            .build()
        )
        insert_resp = await asyncio.to_thread(client.docx.v1.document_block_descendant.create, insert_req)
        if insert_resp.success():
            return f"Document updated: {doc_token} ({deleted} cleared, {len(blocks)} blocks inserted)"
        return f"[error] Block insertion failed: {insert_resp.code} {insert_resp.msg}"
    except Exception as e:
        logger.error("feishu_write_document error: %s", e, exc_info=True)
        return f"[error] {type(e).__name__}: {e}"


@tool
async def feishu_create_calendar_event(
    title: str,
    start_time: str,
    end_time: str,
    description: str = "",
    calendar_id: str = "",
) -> str:
    """Create a Feishu calendar event.

    Args:
        title: Event title.
        start_time: Start time in ISO format (e.g. "2026-04-18T14:00:00+08:00").
        end_time: End time in ISO format.
        description: Event description.
        calendar_id: Calendar ID. If empty, auto-resolves to the primary calendar.
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        # Auto-resolve primary calendar if not specified
        if not calendar_id:
            from src.channels.feishu import _resolve_api_base, _get_tenant_token
            from src.config import load_config

            cfg = load_config()
            domain = cfg.channels.feishu.domain
            api_token = await _get_tenant_token(cfg.channels.feishu.app_id, cfg.channels.feishu.app_secret, domain)
            if not api_token:
                return "[error] Failed to get tenant token"
            api_base = _resolve_api_base(domain)

            import httpx

            async with httpx.AsyncClient(timeout=15) as hc:
                cal_resp = await hc.get(
                    f"{api_base}/calendar/v4/calendars",
                    headers={"Authorization": f"Bearer {api_token}"},
                )
            cal_data = cal_resp.json()
            logger.info(
                "Calendar list response: code=%s items=%s",
                cal_data.get("code"),
                len(cal_data.get("data", {}).get("calendar_list", [])),
            )
            if cal_data.get("code") == 0:
                cal_list = cal_data.get("data", {}).get("calendar_list", [])
                for cal in cal_list:
                    if cal.get("type") == "primary":
                        calendar_id = cal["calendar_id"]
                        break
                if not calendar_id and cal_list:
                    calendar_id = cal_list[0]["calendar_id"]
            if not calendar_id:
                return "[error] No calendar found (bot may need calendar scope). Please specify calendar_id."

        from datetime import datetime
        from lark_oapi.api.calendar.v4 import CreateCalendarEventRequest, CreateCalendarEventRequestBody, TimeStamp

        start_ts = int(datetime.fromisoformat(start_time).timestamp()) * 1000
        end_ts = int(datetime.fromisoformat(end_time).timestamp()) * 1000

        body = (
            CreateCalendarEventRequestBody.builder()
            .summary(title)
            .description(description)
            .start_time(TimeStamp.builder().timestamp(str(start_ts)).build())
            .end_time(TimeStamp.builder().timestamp(str(end_ts)).build())
            .build()
        )
        req = CreateCalendarEventRequest.builder().calendar_id(calendar_id).request_body(body).build()
        resp = await _lark_thread(client.calendar.v4.calendar_event.create, req)
        if resp.success():
            return f"Calendar event created: {title} (calendar: {calendar_id})"
        return f"[error] {resp.code}: {resp.msg}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


def _parse_messages(resp):
    """Extract formatted message list from im.v1.message.list response."""
    if not resp.data or not resp.data.items:
        return None
    messages = []
    for item in resp.data.items:
        msg = getattr(item, "message", None)
        if not msg:
            continue
        body_content = getattr(msg.body, "content", "") if msg.body else ""
        try:
            body = json.loads(body_content) if body_content else {}
            text = body.get("text", body.get("content", str(body)))
        except Exception:
            text = body_content
        sender = getattr(msg, "sender", None)
        sender_id = ""
        if sender and hasattr(sender, "sender_id") and sender.sender_id:
            sender_id = sender.sender_id.open_id
        messages.append(f"[{msg.message_type}] {sender_id}: {text[:200]}")
    return "\n---\n".join(messages) if messages else "No readable messages found"


@tool
async def feishu_get_message_list(chat_id: str, count: int = 20) -> str:
    """Get recent messages from a Feishu chat.

    Args:
        chat_id: The chat ID.
        count: Number of messages to retrieve (max 50).
    """
    from lark_oapi.api.im.v1 import ListMessageRequest

    req = ListMessageRequest.builder().container_id_type("chat").container_id(chat_id).page_size(min(count, 50)).build()
    return await _feishu_call("im.v1.message.list", req, extract=_parse_messages, ok_msg="No messages found")


@tool
async def feishu_recall_message(message_id: str) -> str:
    """Recall (delete) a previously sent message.

    Args:
        message_id: The message ID to recall.
    """
    from lark_oapi.api.im.v1 import DeleteMessageRequest

    req = DeleteMessageRequest.builder().message_id(message_id).build()
    return await _feishu_call("im.v1.message.delete", req, ok_msg=f"Message recalled: {message_id}")


@tool
async def feishu_create_folder(name: str, folder_token: str = "") -> str:
    """Create a folder in Feishu Drive.

    Args:
        name: Folder name.
        folder_token: Parent folder token (empty for root folder).
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        from lark_oapi.api.drive.v1 import CreateFolderFileRequest, CreateFolderFileRequestBody
        from src.channels.feishu import _resolve_api_base, _get_tenant_token
        from src.config import load_config

        # Resolve effective folder token (mirrors original TypeScript createFolder logic)
        effective_token = folder_token if folder_token and folder_token != "0" else ""
        if not effective_token:
            # Try explorer API to get real root folder token
            try:
                cfg = load_config()
                domain = cfg.channels.feishu.domain
                api_token = await _get_tenant_token(cfg.channels.feishu.app_id, cfg.channels.feishu.app_secret, domain)
                if api_token:
                    import httpx

                    api_base = _resolve_api_base(domain)
                    async with httpx.AsyncClient(timeout=10) as hc:
                        r = await hc.get(
                            f"{api_base}/drive/explorer/v2/root_folder/meta",
                            headers={"Authorization": f"Bearer {api_token}"},
                        )
                        if r.status_code == 200:
                            data = r.json()
                            if data.get("code") == 0:
                                root_token = data.get("data", {}).get("token", "")
                                if root_token:
                                    effective_token = root_token
                                    logger.info("Root folder token from explorer API: %s", root_token)
            except Exception as e:
                logger.debug("Explorer API unavailable, using fallback: %s", e)

            if not effective_token:
                effective_token = "0"

        # Use SDK to create folder
        body = CreateFolderFileRequestBody.builder().name(name).folder_token(effective_token).build()
        req = CreateFolderFileRequest.builder().request_body(body).build()
        resp = await _lark_thread(client.drive.v1.file.create_folder, req)

        if resp.success() and resp.data:
            token = resp.data.token or ""
            return f"Folder created: {name} (token={token})"

        code = resp.code
        msg = resp.msg or ""
        logger.warning("create_folder failed: code=%s msg=%s token=%s", code, msg, effective_token)

        if code == 1061002:
            return (
                f"[error] 1061002 params error — folder_token '{effective_token}' may be invalid. "
                f"Please provide a valid parent folder token (starts with 'fldc'). "
                f"Tip: check your app has 'drive:drive' write permission in Feishu Open Platform."
            )
        return f"[error] {code}: {msg}"
    except Exception as e:
        logger.error("feishu_create_folder error: %s", e, exc_info=True)
        return f"[error] {type(e).__name__}: {e}"


@tool
async def feishu_drive_list(folder_token: str = "") -> str:
    """List files and folders in Feishu Drive.

    Args:
        folder_token: Folder token to list (empty for root).
    """
    from lark_oapi.api.drive.v1 import ListFileRequest

    req = ListFileRequest.builder().folder_token(folder_token or "").build()
    return await _feishu_call(
        "drive.v1.file.list", req,
        extract=lambda r: [{"name": f.name, "token": f.token, "type": f.type} for f in r.data.files] if r.data and r.data.files else None,
        ok_msg="No files found",
    )


@tool
async def feishu_send_card(chat_id: str, title: str, content: str) -> str:
    """Send an interactive card message to a Feishu chat.

    Args:
        chat_id: The Feishu chat ID.
        title: Card title.
        content: Card body text (supports markdown).
    """
    from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

    card = json.dumps({
        "schema": "2.0",
        "header": {"title": {"tag": "plain_text", "content": title}},
        "body": {"elements": [{"tag": "markdown", "content": content}]},
    }, ensure_ascii=False)
    body = CreateMessageRequestBody.builder().receive_id(chat_id).msg_type("interactive").content(card).build()
    req = CreateMessageRequest.builder().receive_id_type("chat_id").request_body(body).build()
    return await _feishu_call(
        "im.v1.message.create", req,
        extract=lambda r: f"Card sent to {chat_id}, message_id={r.data.message_id}" if r.data else None,
    )


@tool
async def feishu_create_bitable(name: str, folder_token: str = "") -> str:
    """Create a new Feishu bitable (multi-dimensional table).

    Args:
        name: Bitable name.
        folder_token: Parent folder token (empty for root).
    """
    from lark_oapi.api.bitable.v1 import CreateAppRequest, ReqApp

    body = ReqApp.builder().name(name).folder_token(folder_token or "").build()
    req = CreateAppRequest.builder().request_body(body).build()
    return await _feishu_call(
        "bitable.v1.app.create", req,
        extract=lambda r: f"Bitable created: {r.data.app.name} (app_token={r.data.app.app_token})" if r.data and r.data.app else None,
    )


@tool
async def feishu_bitable_list_records(app_token: str, table_id: str, page_size: int = 20) -> str:
    """List records from a Feishu bitable table.

    Args:
        app_token: Bitable app token.
        table_id: Table ID within the bitable.
        page_size: Max records to fetch (max 500).
    """
    from lark_oapi.api.bitable.v1 import ListAppTableRecordRequest

    req = ListAppTableRecordRequest.builder().app_token(app_token).table_id(table_id).page_size(min(page_size, 500)).build()
    return await _feishu_call(
        "bitable.v1.app_table_record.list", req,
        extract=lambda r: [rec.fields if hasattr(rec, "fields") else str(rec) for rec in r.data.items] if r.data and r.data.items else None,
        ok_msg="No records found",
    )


@tool
async def feishu_bitable_add_record(app_token: str, table_id: str, fields_json: str) -> str:
    """Add a record to a Feishu bitable table.

    Args:
        app_token: Bitable app token.
        table_id: Table ID.
        fields_json: Record fields as JSON object, e.g. '{"Name": "test", "Age": 25}'.
    """
    from lark_oapi.api.bitable.v1 import CreateAppTableRecordRequest, AppTableRecord

    fields = json.loads(fields_json)
    body = AppTableRecord.builder().fields(fields).build()
    req = CreateAppTableRecordRequest.builder().app_token(app_token).table_id(table_id).request_body(body).build()
    return await _feishu_call(
        "bitable.v1.app_table_record.create", req,
        extract=lambda r: f"Record added (record_id={r.data.record.record_id})" if r.data and r.data.record else None,
    )


# ============================================================
# Document advanced operations (feishu_doc supplement)
# ============================================================


@tool
async def feishu_doc_append(doc_token: str, content: str) -> str:
    """Append content to the end of a Feishu document.

    Args:
        doc_token: Document token.
        content: Markdown content to append.
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        from lark_oapi.api.docx.v1 import (
            ConvertDocumentRequest,
            ConvertDocumentRequestBody,
            CreateDocumentBlockDescendantRequest,
            CreateDocumentBlockDescendantRequestBody,
        )

        convert_body = ConvertDocumentRequestBody.builder().content_type("markdown").content(content).build()
        convert_req = ConvertDocumentRequest.builder().request_body(convert_body).build()
        convert_resp = await _lark_thread(client.docx.v1.document.convert, convert_req)
        if not convert_resp.success() or not convert_resp.data:
            return f"[error] Markdown conversion failed: {convert_resp.code} {convert_resp.msg}"

        blocks = convert_resp.data.blocks or []
        first_level_ids = convert_resp.data.first_level_block_ids or []
        if not blocks or not first_level_ids:
            return "[error] Conversion produced no blocks"

        insert_body = (
            CreateDocumentBlockDescendantRequestBody.builder()
            .children_id(first_level_ids)
            .index(-1)
            .descendants(blocks)
            .build()
        )
        insert_req = (
            CreateDocumentBlockDescendantRequest.builder()
            .document_id(doc_token)
            .block_id(doc_token)
            .request_body(insert_body)
            .build()
        )
        insert_resp = await _lark_thread(client.docx.v1.document_block_descendant.create, insert_req)
        if insert_resp.success():
            return f"Appended {len(blocks)} blocks to {doc_token}"
        return f"[error] Block insertion failed: {insert_resp.code} {insert_resp.msg}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


@tool
async def feishu_doc_insert(doc_token: str, content: str, after_block_id: str) -> str:
    """Insert content after a specific block in a Feishu document.

    Args:
        doc_token: Document token.
        content: Markdown content to insert.
        after_block_id: Block ID to insert after.
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        from lark_oapi.api.docx.v1 import (
            GetDocumentBlockChildrenRequest,
            ConvertDocumentRequest,
            ConvertDocumentRequestBody,
            CreateDocumentBlockDescendantRequest,
            CreateDocumentBlockDescendantRequestBody,
        )

        # Find insertion index by listing children
        children_req = (
            GetDocumentBlockChildrenRequest.builder().document_id(doc_token).block_id(doc_token).page_size(500).build()
        )
        children_resp = await _lark_thread(client.docx.v1.document_block_children.get, children_req)

        insert_index = -1  # default: append at end
        if children_resp.success() and children_resp.data and children_resp.data.items:
            for i, item in enumerate(children_resp.data.items):
                if item.block_id == after_block_id:
                    insert_index = i + 1
                    break

        convert_body = ConvertDocumentRequestBody.builder().content_type("markdown").content(content).build()
        convert_req = ConvertDocumentRequest.builder().request_body(convert_body).build()
        convert_resp = await _lark_thread(client.docx.v1.document.convert, convert_req)
        if not convert_resp.success() or not convert_resp.data:
            return f"[error] Markdown conversion failed: {convert_resp.code} {convert_resp.msg}"

        blocks = convert_resp.data.blocks or []
        first_level_ids = convert_resp.data.first_level_block_ids or []
        if not blocks or not first_level_ids:
            return "[error] Conversion produced no blocks"

        insert_body = (
            CreateDocumentBlockDescendantRequestBody.builder()
            .children_id(first_level_ids)
            .index(insert_index)
            .descendants(blocks)
            .build()
        )
        insert_req = (
            CreateDocumentBlockDescendantRequest.builder()
            .document_id(doc_token)
            .block_id(doc_token)
            .request_body(insert_body)
            .build()
        )
        insert_resp = await _lark_thread(client.docx.v1.document_block_descendant.create, insert_req)
        if insert_resp.success():
            return f"Inserted {len(blocks)} blocks after {after_block_id} (index={insert_index})"
        return f"[error] Block insertion failed: {insert_resp.code} {insert_resp.msg}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


@tool
async def feishu_doc_list_blocks(doc_token: str) -> str:
    """List all blocks in a Feishu document.

    Args:
        doc_token: Document token.
    """
    from lark_oapi.api.docx.v1 import ListDocumentBlockRequest

    req = ListDocumentBlockRequest.builder().document_id(doc_token).page_size(500).build()
    return await _feishu_call(
        "docx.v1.document_block.list", req,
        extract=lambda r: [{"block_id": b.block_id, "block_type": b.block_type, "parent_id": b.parent_id} for b in r.data.items] if r.data and r.data.items else None,
        ok_msg="No blocks found",
    )


@tool
async def feishu_doc_get_block(doc_token: str, block_id: str) -> str:
    """Get details of a specific block in a Feishu document.

    Args:
        doc_token: Document token.
        block_id: Block ID to retrieve.
    """
    from lark_oapi.api.docx.v1 import GetDocumentBlockRequest

    req = GetDocumentBlockRequest.builder().document_id(doc_token).block_id(block_id).build()
    return await _feishu_call(
        "docx.v1.document_block.get", req,
        extract=lambda r: {
            "block_id": r.data.block.block_id, "block_type": r.data.block.block_type,
            "parent_id": r.data.block.parent_id, "children": r.data.block.children,
        } if r.data and r.data.block else f"[error] Block not found: {block_id}",
    )


@tool
async def feishu_doc_update_block(doc_token: str, block_id: str, content: str) -> str:
    """Update the text content of a block in a Feishu document.

    Args:
        doc_token: Document token.
        block_id: Block ID to update.
        content: New text content.
    """
    from lark_oapi.api.docx.v1 import (
        PatchDocumentBlockRequest, UpdateBlockRequest, UpdateTextElementsRequest, TextElement, TextRun,
    )

    text_elements = [TextElement.builder().text_run(TextRun.builder().content(content).build()).build()]
    update_body = (
        UpdateBlockRequest.builder().block_id(block_id)
        .update_text_elements(UpdateTextElementsRequest.builder().elements(text_elements).build()).build()
    )
    req = PatchDocumentBlockRequest.builder().document_id(doc_token).block_id(block_id).request_body(update_body).build()
    return await _feishu_call("docx.v1.document_block.patch", req, ok_msg=f"Block updated: {block_id}")


@tool
async def feishu_doc_delete_block(doc_token: str, block_id: str) -> str:
    """Delete a block from a Feishu document.

    Args:
        doc_token: Document token.
        block_id: Block ID to delete.
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        from lark_oapi.api.docx.v1 import (
            GetDocumentBlockChildrenRequest,
            BatchDeleteDocumentBlockChildrenRequest,
            BatchDeleteDocumentBlockChildrenRequestBody,
        )

        # Search for block across all levels (root children first, then sub-blocks)
        async def _find_block(root_block_id: str) -> tuple:
            """Return (parent_id, index) or (None, -1)."""
            children_req = (
                GetDocumentBlockChildrenRequest.builder()
                .document_id(doc_token)
                .block_id(root_block_id)
                .page_size(500)
                .build()
            )
            children_resp = await _lark_thread(client.docx.v1.document_block_children.get, children_req)
            if not children_resp.success() or not children_resp.data or not children_resp.data.items:
                return None, -1
            for i, item in enumerate(children_resp.data.items):
                if item.block_id == block_id:
                    return root_block_id, i
            # Recurse into child blocks that can contain children
            container_types = {1, 2, 3, 13, 14, 15, 17, 18, 19, 22, 23, 24}  # page, text, heading, list, table, etc.
            for item in children_resp.data.items:
                bt = getattr(item, "block_type", 0)
                if bt in container_types:
                    pid, idx = await _find_block(item.block_id)
                    if pid is not None:
                        return pid, idx
            return None, -1

        parent_id, target_index = await _find_block(doc_token)
        if parent_id is None or target_index < 0:
            return f"[error] Block {block_id} not found in document"

        # end_index is exclusive in Lark API
        del_body = (
            BatchDeleteDocumentBlockChildrenRequestBody.builder()
            .start_index(target_index)
            .end_index(target_index + 1)
            .build()
        )
        del_req = (
            BatchDeleteDocumentBlockChildrenRequest.builder()
            .document_id(doc_token)
            .block_id(parent_id)
            .request_body(del_body)
            .build()
        )
        del_resp = await _lark_thread(client.docx.v1.document_block_children.batch_delete, del_req)
        if del_resp.success():
            return f"Block deleted: {block_id}"
        return f"[error] {del_resp.code}: {del_resp.msg}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


@tool
async def feishu_doc_create_table(doc_token: str, row_size: int, column_size: int) -> str:
    """Create a table in a Feishu document.

    Args:
        doc_token: Document token.
        row_size: Number of rows.
        column_size: Number of columns.

    Note: The Feishu Open API does not support creating table blocks directly via
    create_block_children (always returns 1770001). Use feishu_doc_append with
    markdown table syntax as a workaround, or create tables manually in the document.
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        from lark_oapi.api.docx.v1 import (
            CreateDocumentBlockChildrenRequest,
            CreateDocumentBlockChildrenRequestBody,
            Block,
            Table,
            TableProperty,
        )

        cells = [""] * (row_size * column_size)
        prop = TableProperty.builder().row_size(row_size).column_size(column_size).build()
        table = Table.builder().cells(cells).property(prop).build()
        block = Block.builder().block_type(17).table(table).build()

        body = CreateDocumentBlockChildrenRequestBody.builder().children([block]).index(-1).build()
        req = (
            CreateDocumentBlockChildrenRequest.builder()
            .document_id(doc_token)
            .block_id(doc_token)
            .request_body(body)
            .build()
        )
        resp = await _lark_thread(client.docx.v1.document_block_children.create, req)
        if resp.success() and resp.data and resp.data.blocks:
            return f"Table created: {resp.data.blocks[0].block_id} ({row_size}x{column_size})"
        return (
            f"[error] Table creation not supported by current API scope (code={resp.code}). "
            "Use feishu_doc_append with markdown table syntax instead."
        )
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


@tool
async def feishu_doc_insert_table_row(doc_token: str, table_block_id: str, row_index: int = -1) -> str:
    """Insert a row into a table in a Feishu document.

    Args:
        doc_token: Document token.
        table_block_id: Table block ID.
        row_index: Row index to insert at (-1 for end).
    """
    from lark_oapi.api.docx.v1 import PatchDocumentBlockRequest, UpdateBlockRequest, InsertTableRowRequest

    body = UpdateBlockRequest.builder().block_id(table_block_id).insert_table_row(InsertTableRowRequest.builder().row_index(row_index).build()).build()
    req = PatchDocumentBlockRequest.builder().document_id(doc_token).block_id(table_block_id).request_body(body).build()
    return await _feishu_call("docx.v1.document_block.patch", req, ok_msg=f"Row inserted at index {row_index}")


@tool
async def feishu_doc_insert_table_col(doc_token: str, table_block_id: str, col_index: int = -1) -> str:
    """Insert a column into a table in a Feishu document.

    Args:
        doc_token: Document token.
        table_block_id: Table block ID.
        col_index: Column index to insert at (-1 for end).
    """
    from lark_oapi.api.docx.v1 import PatchDocumentBlockRequest, UpdateBlockRequest, InsertTableColumnRequest

    body = UpdateBlockRequest.builder().block_id(table_block_id).insert_table_column(InsertTableColumnRequest.builder().column_index(col_index).build()).build()
    req = PatchDocumentBlockRequest.builder().document_id(doc_token).block_id(table_block_id).request_body(body).build()
    return await _feishu_call("docx.v1.document_block.patch", req, ok_msg=f"Column inserted at index {col_index}")


@tool
async def feishu_doc_delete_table_rows(doc_token: str, table_block_id: str, row_start: int, row_end: int) -> str:
    """Delete rows from a table in a Feishu document.

    Args:
        doc_token: Document token.
        table_block_id: Table block ID.
        row_start: Start row index (0-based).
        row_end: End row index (inclusive).
    """
    from lark_oapi.api.docx.v1 import PatchDocumentBlockRequest, UpdateBlockRequest, DeleteTableRowsRequest

    body = UpdateBlockRequest.builder().block_id(table_block_id).delete_table_rows(
        DeleteTableRowsRequest.builder().row_start_index(row_start).row_end_index(row_end).build()
    ).build()
    req = PatchDocumentBlockRequest.builder().document_id(doc_token).block_id(table_block_id).request_body(body).build()
    return await _feishu_call("docx.v1.document_block.patch", req, ok_msg=f"Rows deleted: {row_start}-{row_end}")


@tool
async def feishu_doc_delete_table_cols(doc_token: str, table_block_id: str, col_start: int, col_end: int) -> str:
    """Delete columns from a table in a Feishu document.

    Args:
        doc_token: Document token.
        table_block_id: Table block ID.
        col_start: Start column index (0-based).
        col_end: End column index (inclusive).
    """
    from lark_oapi.api.docx.v1 import PatchDocumentBlockRequest, UpdateBlockRequest, DeleteTableColumnsRequest

    body = UpdateBlockRequest.builder().block_id(table_block_id).delete_table_columns(
        DeleteTableColumnsRequest.builder().column_start_index(col_start).column_end_index(col_end).build()
    ).build()
    req = PatchDocumentBlockRequest.builder().document_id(doc_token).block_id(table_block_id).request_body(body).build()
    return await _feishu_call("docx.v1.document_block.patch", req, ok_msg=f"Columns deleted: {col_start}-{col_end}")


# ============================================================
# Drive operations supplement
# ============================================================


@tool
async def feishu_drive_info(file_token: str) -> str:
    """Get information about a file or folder in Feishu Drive.

    Args:
        file_token: File or folder token.
    """
    from lark_oapi.api.drive.v1 import ListFileRequest

    req = ListFileRequest.builder().folder_token("").page_size(100).build()
    return await _feishu_call(
        "drive.v1.file.list", req,
        extract=lambda r: next(
            ({"token": f.token, "name": f.name, "type": f.type, "url": getattr(f, "url", "")}
             for f in r.data.files if f.token == file_token),
            f"[error] File {file_token} not found in root folder listing",
        ) if r.data and r.data.files else None,
    )


@tool
async def feishu_drive_move(file_token: str, folder_token: str, file_type: str = "") -> str:
    """Move a file or folder to another folder in Feishu Drive.

    Args:
        file_token: File token to move.
        folder_token: Target folder token (empty for root).
        file_type: File type (docx, sheet, bitable, folder, file, etc). Auto-detected if empty.
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        from lark_oapi.api.drive.v1 import MoveFileRequest, MoveFileRequestBody

        # Auto-detect file_type if not specified
        if not file_type:
            from src.channels.feishu import _resolve_api_base, _get_tenant_token
            from src.config import load_config

            cfg = load_config()
            api_token = await _get_tenant_token(
                cfg.channels.feishu.app_id, cfg.channels.feishu.app_secret, cfg.channels.feishu.domain
            )
            if api_token:
                import httpx

                api_base = _resolve_api_base(cfg.channels.feishu.domain)
                async with httpx.AsyncClient(timeout=10) as hc:
                    r = await hc.get(
                        f"{api_base}/drive/v1/files/{file_token}",
                        headers={"Authorization": f"Bearer {api_token}"},
                    )
                if r.status_code == 200:
                    file_data = r.json()
                    if file_data.get("code") == 0:
                        file_type = file_data.get("data", {}).get("type", "")
            if not file_type:
                file_type = "docx"  # fallback

        body = MoveFileRequestBody.builder().type(file_type).folder_token(folder_token).build()
        req = MoveFileRequest.builder().file_token(file_token).request_body(body).build()
        resp = await _lark_thread(client.drive.v1.file.move, req)
        if resp.success():
            return f"File moved: {file_token} -> {folder_token}"
        return f"[error] {resp.code}: {resp.msg}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


@tool
async def feishu_drive_delete(file_token: str, file_type: str) -> str:
    """Delete a file or folder from Feishu Drive.

    Args:
        file_token: File token to delete.
        file_type: File type (docx, sheet, bitable, folder, file, etc).
    """
    from lark_oapi.api.drive.v1 import DeleteFileRequest

    req = DeleteFileRequest.builder().file_token(file_token).type(file_type).build()
    return await _feishu_call("drive.v1.file.delete", req, ok_msg=f"File deleted: {file_token}")


def _parse_comments(resp):
    """Extract comment list from drive.v1.file_comment.list response."""
    if not resp.data or not resp.data.items:
        return None
    comments = []
    for c in resp.data.items:
        content_str = ""
        reply_list = getattr(c, "reply_list", None)
        if reply_list:
            for reply in getattr(reply_list, "replies", []) or []:
                rc = getattr(reply, "content", None)
                if rc:
                    for elem in getattr(rc, "elements", []) or []:
                        tr = getattr(elem, "text_run", None)
                        if tr:
                            content_str += getattr(tr, "text", "")
        if not content_str:
            body_obj = getattr(c, "body", None)
            if body_obj and hasattr(body_obj, "content") and isinstance(body_obj.content, list):
                for p in body_obj.content:
                    if hasattr(p, "text"):
                        content_str += getattr(p, "text", "")
        created_by = getattr(c, "created_by", None)
        comments.append({
            "comment_id": getattr(c, "comment_id", ""),
            "content": content_str,
            "user_id": getattr(created_by, "user_id", "") if created_by else "",
        })
    return comments


@tool
async def feishu_drive_list_comments(file_token: str, file_type: str = "docx", page_size: int = 20) -> str:
    """List comments on a file in Feishu Drive.

    Args:
        file_token: File token.
        file_type: File type (docx, sheet, bitable, file, etc).
        page_size: Max comments to fetch.
    """
    from lark_oapi.api.drive.v1 import ListFileCommentRequest

    req = ListFileCommentRequest.builder().file_token(file_token).file_type(file_type).page_size(min(page_size, 100)).build()
    return await _feishu_call("drive.v1.file_comment.list", req, extract=_parse_comments, ok_msg="No comments found")


@tool
async def feishu_drive_add_comment(file_token: str, content: str, file_type: str = "docx", block_id: str = "") -> str:
    """Add a comment to a file in Feishu Drive.

    Args:
        file_token: File token.
        content: Comment text content.
        file_type: File type (docx, sheet, etc).
        block_id: Block ID for docx anchor (optional).
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        from src.channels.feishu import _resolve_api_base, _get_tenant_token
        from src.config import load_config

        cfg = load_config()
        domain = cfg.channels.feishu.domain
        api_token = await _get_tenant_token(cfg.channels.feishu.app_id, cfg.channels.feishu.app_secret, domain)
        if not api_token:
            return "[error] Failed to get tenant token"
        api_base = _resolve_api_base(domain)

        # Build the comment body with required reply_list structure
        body_data = {
            "reply_list": {
                "replies": [{"content": {"elements": [{"type": "text_run", "text_run": {"text": content}}]}}]
            }
        }
        if block_id:
            body_data["block_id"] = block_id

        import httpx

        async with httpx.AsyncClient(timeout=15) as hc:
            r = await hc.post(
                f"{api_base}/drive/v1/files/{file_token}/comments",
                headers={
                    "Authorization": f"Bearer {api_token}",
                    "Content-Type": "application/json; charset=utf-8",
                },
                params={"file_type": file_type, "user_id_type": "open_id"},
                json=body_data,
            )
        data = r.json()
        if data.get("code") == 0:
            comment_id = data.get("data", {}).get("comment_id", "")
            return f"Comment added to {file_token}" + (
                f" (id: {comment_id})" if comment_id else "(no comment_id returned)"
            )
        return f"[error] {data.get('code', r.status_code)}: {data.get('msg', '')}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


@tool
async def feishu_drive_reply_comment(file_token: str, comment_id: str, content: str, file_type: str = "docx") -> str:
    """Reply to a comment on a file in Feishu Drive.

    Args:
        file_token: File token.
        comment_id: Comment ID to reply to.
        content: Reply text content.
        file_type: File type (docx, sheet, etc).
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        import httpx
        from src.channels.feishu import _resolve_api_base, _get_tenant_token
        from src.config import load_config

        cfg = load_config()
        domain = cfg.channels.feishu.domain
        api_token = await _get_tenant_token(cfg.channels.feishu.app_id, cfg.channels.feishu.app_secret, domain)
        if not api_token:
            return "[error] Failed to get tenant token"
        api_base = _resolve_api_base(domain)

        async with httpx.AsyncClient(timeout=15) as hc:
            r = await hc.post(
                f"{api_base}/drive/v1/files/{file_token}/comments/{comment_id}/replies",
                headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json; charset=utf-8"},
                params={"file_type": file_type},
                json={"content": {"elements": [{"type": "text_run", "text_run": {"text": content}}]}},
            )
        data = r.json()
        if data.get("code") == 0:
            return f"Reply added to comment {comment_id}"
        code = data.get("code", r.status_code)
        msg = data.get("msg", "")
        if code == 1069302:
            return f"[error] 1069302: 该文档的评论已禁用回复功能，请在飞书文档评论设置中开启回复权限"
        return f"[error] {code}: {msg}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


# ============================================================
# Bitable supplement
# ============================================================


@tool
async def feishu_bitable_get_meta(url: str) -> str:
    """Parse a Bitable URL and get app_token, table_id, and table list.

    Args:
        url: Bitable URL (/base/XXX or /wiki/XXX with optional ?table=YYY).
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        from urllib.parse import urlparse, parse_qs

        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")
        table_param = parse_qs(parsed.query).get("table", [None])[0]

        app_token = ""
        obj_token = ""

        if "base" in path_parts:
            idx = path_parts.index("base")
            if idx + 1 < len(path_parts):
                app_token = path_parts[idx + 1]
        elif "wiki" in path_parts:
            # Wiki URL: need to resolve node to get obj_token
            idx = path_parts.index("wiki")
            if idx + 1 < len(path_parts):
                node_token = path_parts[idx + 1]
                from lark_oapi.api.wiki.v2 import GetNodeSpaceRequest

                node_req = GetNodeSpaceRequest.builder().token(node_token).build()
                node_resp = await _lark_thread(client.wiki.v2.space.get_node, node_req)
                if node_resp.success() and node_resp.data and node_resp.data.node:
                    obj_token = node_resp.data.node.obj_token or ""
                    app_token = obj_token
        else:
            return f"[error] Cannot parse URL: {url}"

        if not app_token:
            return f"[error] Could not extract app_token from URL: {url}"

        # Get table list
        from lark_oapi.api.bitable.v1 import ListAppTableRequest

        table_req = ListAppTableRequest.builder().app_token(app_token).build()
        table_resp = await _lark_thread(client.bitable.v1.app_table.list, table_req)

        tables = []
        table_id = table_param or ""
        if table_resp.success() and table_resp.data and table_resp.data.items:
            for t in table_resp.data.items:
                tables.append({"table_id": t.table_id, "name": t.name})
                if not table_id:
                    table_id = t.table_id

        result = {
            "app_token": app_token,
            "table_id": table_id,
            "tables": tables,
        }
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


@tool
async def feishu_bitable_list_fields(app_token: str, table_id: str) -> str:
    """List all fields (columns) in a Bitable table.

    Args:
        app_token: Bitable app token.
        table_id: Table ID.
    """
    from lark_oapi.api.bitable.v1 import ListAppTableFieldRequest

    req = ListAppTableFieldRequest.builder().app_token(app_token).table_id(table_id).build()
    return await _feishu_call(
        "bitable.v1.app_table_field.list", req,
        extract=lambda r: [{"field_id": f.field_id, "field_name": f.field_name, "type": f.type} for f in r.data.items] if r.data and r.data.items else None,
        ok_msg="No fields found",
    )


@tool
async def feishu_bitable_get_record(app_token: str, table_id: str, record_id: str) -> str:
    """Get a single record from a Bitable table.

    Args:
        app_token: Bitable app token.
        table_id: Table ID.
        record_id: Record ID to retrieve.
    """
    from lark_oapi.api.bitable.v1 import GetAppTableRecordRequest

    req = GetAppTableRecordRequest.builder().app_token(app_token).table_id(table_id).record_id(record_id).build()
    return await _feishu_call(
        "bitable.v1.app_table_record.get", req,
        extract=lambda r: {
            "record_id": r.data.record.record_id,
            "fields": r.data.record.fields if hasattr(r.data.record, "fields") else {},
        } if r.data and r.data.record else f"[error] Record not found: {record_id}",
    )


@tool
async def feishu_bitable_update_record(app_token: str, table_id: str, record_id: str, fields_json: str) -> str:
    """Update an existing record in a Bitable table.

    Args:
        app_token: Bitable app token.
        table_id: Table ID.
        record_id: Record ID to update.
        fields_json: Updated fields as JSON object, e.g. '{"Name": "new value"}'.
    """
    from lark_oapi.api.bitable.v1 import UpdateAppTableRecordRequest, AppTableRecord

    fields = json.loads(fields_json)
    record = AppTableRecord.builder().fields(fields).build()
    req = UpdateAppTableRecordRequest.builder().app_token(app_token).table_id(table_id).record_id(record_id).request_body(record).build()
    return await _feishu_call(
        "bitable.v1.app_table_record.update", req,
        extract=lambda r: f"Record updated: {record_id}" if r.data and r.data.record else None,
    )


@tool
async def feishu_bitable_create_field(app_token: str, table_id: str, field_name: str, field_type: int) -> str:
    """Create a new field (column) in a Bitable table.

    Args:
        app_token: Bitable app token.
        table_id: Table ID.
        field_name: Field name.
        field_type: Field type number (1=Text, 2=Number, 3=SingleSelect, 4=MultiSelect, 5=Date, 7=Checkbox, 11=Phone, 13=URL, 15=Email, 1001=CreatedTime, 1002=ModifiedTime, 1003=CreatedBy, 1004=LastModifiedBy).
    """
    from lark_oapi.api.bitable.v1 import CreateAppTableFieldRequest, AppTableField

    field_obj = AppTableField.builder().field_name(field_name).type(field_type).build()
    req = CreateAppTableFieldRequest.builder().app_token(app_token).table_id(table_id).request_body(field_obj).build()
    return await _feishu_call(
        "bitable.v1.app_table_field.create", req,
        extract=lambda r: f"Field created: {field_name} (field_id={r.data.field.field_id})" if r.data and r.data.field else None,
    )


# ============================================================
# Wiki operations
# ============================================================


@tool
async def feishu_wiki_list_spaces() -> str:
    """List all wiki knowledge spaces.

    Args: (none)
    """
    from lark_oapi.api.wiki.v2 import ListSpaceRequest

    req = ListSpaceRequest.builder().page_size(50).build()
    return await _feishu_call(
        "wiki.v2.space.list", req,
        extract=lambda r: [{"space_id": getattr(s, "space_id", ""), "name": getattr(s, "name", ""), "description": getattr(s, "description", "")} for s in r.data.items] if r.data and r.data.items else None,
        ok_msg="No wiki spaces found",
    )


@tool
async def feishu_wiki_list_nodes(space_id: str, parent_node_token: str = "") -> str:
    """List nodes in a wiki space.

    Args:
        space_id: Wiki space ID.
        parent_node_token: Parent node token (empty for root).
    """
    from lark_oapi.api.wiki.v2 import ListSpaceNodeRequest

    req = ListSpaceNodeRequest.builder().space_id(space_id).parent_node_token(parent_node_token).page_size(50).build()
    return await _feishu_call(
        "wiki.v2.space_node.list", req,
        extract=lambda r: [{"node_token": getattr(n, "node_token", ""), "title": getattr(n, "title", ""), "obj_type": getattr(n, "obj_type", ""), "obj_token": getattr(n, "obj_token", "")} for n in r.data.items] if r.data and r.data.items else None,
        ok_msg="No nodes found",
    )


@tool
async def feishu_wiki_get_node(token: str) -> str:
    """Get details of a wiki node.

    Args:
        token: Wiki node token.
    """
    from lark_oapi.api.wiki.v2 import GetNodeSpaceRequest

    req = GetNodeSpaceRequest.builder().token(token).build()
    return await _feishu_call(
        "wiki.v2.space.get_node", req,
        extract=lambda r: {
            "node_token": getattr(r.data.node, "node_token", ""), "space_id": getattr(r.data.node, "space_id", ""),
            "title": getattr(r.data.node, "title", ""), "obj_type": getattr(r.data.node, "obj_type", ""),
            "obj_token": getattr(r.data.node, "obj_token", ""), "parent_node_token": getattr(r.data.node, "parent_node_token", ""),
        } if r.data and r.data.node else f"[error] Node not found: {token}",
    )


@tool
async def feishu_wiki_create_node(
    space_id: str, title: str, obj_type: str = "docx", parent_node_token: str = ""
) -> str:
    """Create a new node in a wiki space.

    Args:
        space_id: Wiki space ID.
        title: Node title.
        obj_type: Object type (docx, sheet, bitable, mindnote, file).
        parent_node_token: Parent node token (empty for root).
    """
    from lark_oapi.api.wiki.v2 import CreateSpaceNodeRequest, Node

    body = Node.builder().title(title).obj_type(obj_type).parent_node_token(parent_node_token).build()
    req = CreateSpaceNodeRequest.builder().space_id(space_id).request_body(body).build()
    return await _feishu_call(
        "wiki.v2.space_node.create", req,
        extract=lambda r: f"Wiki node created: {r.data.node.title} (token={r.data.node.node_token}, obj_token={r.data.node.obj_token})" if r.data and r.data.node else None,
    )


@tool
async def feishu_wiki_move_node(
    space_id: str, node_token: str, target_space_id: str = "", target_parent_token: str = ""
) -> str:
    """Move a wiki node to another location.

    Args:
        space_id: Source space ID.
        node_token: Node token to move.
        target_space_id: Target space ID (empty = same space).
        target_parent_token: Target parent node token.
    """
    from lark_oapi.api.wiki.v2 import MoveSpaceNodeRequest, MoveSpaceNodeRequestBody

    body = MoveSpaceNodeRequestBody.builder().target_space_id(target_space_id or space_id).target_parent_token(target_parent_token).build()
    req = MoveSpaceNodeRequest.builder().space_id(space_id).node_token(node_token).request_body(body).build()
    return await _feishu_call("wiki.v2.space_node.move", req, ok_msg=f"Wiki node moved: {node_token}")


@tool
async def feishu_wiki_rename_node(space_id: str, node_token: str, title: str) -> str:
    """Rename a wiki node.

    Args:
        space_id: Wiki space ID.
        node_token: Node token to rename.
        title: New title.
    """
    from lark_oapi.api.wiki.v2 import UpdateTitleSpaceNodeRequest, UpdateTitleSpaceNodeRequestBody

    body = UpdateTitleSpaceNodeRequestBody.builder().title(title).build()
    req = UpdateTitleSpaceNodeRequest.builder().space_id(space_id).node_token(node_token).request_body(body).build()
    return await _feishu_call("wiki.v2.space_node.update_title", req, ok_msg=f"Wiki node renamed: {node_token} -> {title}")


# ============================================================
# Permission management
# ============================================================


@tool
async def feishu_perm_list_members(token: str, perm_type: str = "docx", page_size: int = 50) -> str:
    """List members with permissions on a resource.

    Args:
        token: Resource token (document, sheet, etc).
        perm_type: Resource type (docx, sheet, bitable, file, wiki, etc).
        page_size: Max members to fetch.
    """
    from lark_oapi.api.drive.v1 import ListPermissionMemberRequest

    req = ListPermissionMemberRequest.builder().token(token).type(perm_type).build()
    return await _feishu_call(
        "drive.v1.permission_member.list", req,
        extract=lambda r: [{"member_id": getattr(m, "member_id", ""), "member_type": getattr(m, "member_type", ""), "perm": getattr(m, "perm", ""), "name": getattr(m, "name", "")} for m in r.data.items] if r.data and r.data.items else None,
        ok_msg="No members found",
    )


@tool
async def feishu_perm_add_member(
    token: str, perm_type: str = "docx", member_type: str = "openid", member_id: str = "", perm: str = "read_only"
) -> str:
    """Add a member with permissions to a resource.

    Args:
        token: Resource token.
        perm_type: Resource type (docx, sheet, bitable, file, wiki, etc).
        member_type: Member type (openid, unionid, email, openchat, userid, etc).
        member_id: Member ID value.
        perm: Permission level (read_only, edit, full_access).
    """
    from lark_oapi.api.drive.v1 import CreatePermissionMemberRequest
    from lark_oapi.api.drive.v1.model import BaseMember

    body = BaseMember.builder().member_type(member_type).member_id(member_id).perm(perm).build()
    req = CreatePermissionMemberRequest.builder().token(token).type(perm_type).need_notification(True).request_body(body).build()
    return await _feishu_call(
        "drive.v1.permission_member.create", req,
        ok_msg=f"Permission granted: {member_type}:{member_id} = {perm} on {token}",
    )


@tool
async def feishu_perm_remove_member(token: str, perm_type: str, member_type: str, member_id: str) -> str:
    """Remove a member's permissions from a resource.

    Args:
        token: Resource token.
        perm_type: Resource type (docx, sheet, bitable, file, wiki, etc).
        member_type: Member type (openid, unionid, email, openchat, userid, etc).
        member_id: Member ID to remove.
    """
    from lark_oapi.api.drive.v1 import DeletePermissionMemberRequest

    req = DeletePermissionMemberRequest.builder().token(token).type(perm_type).member_type(member_type).member_id(member_id).build()
    return await _feishu_call(
        "drive.v1.permission_member.delete", req,
        ok_msg=f"Permission removed: {member_type}:{member_id} from {token}",
    )


# ============================================================
# Chat supplement
# ============================================================


@tool
async def feishu_get_member_info(member_id: str, id_type: str = "open_id") -> str:
    """Get detailed info about a Feishu user.

    Args:
        member_id: User ID (open_id, union_id, or user_id).
        id_type: ID type (open_id, union_id, user_id).
    """
    from lark_oapi.api.contact.v3 import GetUserRequest

    req = GetUserRequest.builder().user_id(member_id).user_id_type(id_type).build()
    return await _feishu_call(
        "contact.v3.user.get", req,
        extract=lambda r: {
            "open_id": getattr(r.data.user, "open_id", ""), "union_id": getattr(r.data.user, "union_id", ""),
            "user_id": getattr(r.data.user, "user_id", ""), "name": getattr(r.data.user, "name", ""),
            "en_name": getattr(r.data.user, "en_name", ""), "email": getattr(r.data.user, "email", ""),
            "mobile": getattr(r.data.user, "mobile", ""),
        } if r.data and r.data.user else f"[error] User not found: {member_id}",
    )


# ============================================================
# Document image upload
# ============================================================


@tool
async def feishu_doc_upload_image(doc_token: str, file_path: str) -> str:
    """Upload a local image and insert it into a Feishu document.

    Args:
        doc_token: Document token.
        file_path: Local path to the image file.
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        import os
        from lark_oapi.api.drive.v1 import UploadAllMediaRequest, UploadAllMediaRequestBody
        from lark_oapi.api.docx.v1 import (
            CreateDocumentBlockChildrenRequest,
            CreateDocumentBlockChildrenRequestBody,
            Block,
            Image,
        )

        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            return f"[error] File not found: {abs_path}"

        file_name = os.path.basename(abs_path)
        file_size = os.path.getsize(abs_path)

        # Upload image to drive
        def _do_upload():
            with open(abs_path, "rb") as f:
                body = (
                    UploadAllMediaRequestBody.builder()
                    .file_name(file_name)
                    .parent_type("docx_image")
                    .parent_node(doc_token)
                    .size(file_size)
                    .file(f)
                    .build()
                )
                req = UploadAllMediaRequest.builder().request_body(body).build()
                return client.drive.v1.media.upload_all(req)

        upload_resp = await asyncio.to_thread(_do_upload)
        if not upload_resp.success():
            return f"[error] Image upload failed: {upload_resp.code} {upload_resp.msg}"

        file_token = ""
        if upload_resp.data:
            file_token = getattr(upload_resp.data, "file_token", "")

        if not file_token:
            return "[error] Upload succeeded but no file_token returned"

        # Insert image block
        image = Image.builder().token(file_token).build()
        block = Block.builder().block_type(27).image(image).build()  # 27 = image

        body = CreateDocumentBlockChildrenRequestBody.builder().children([block]).index(-1).build()
        req = (
            CreateDocumentBlockChildrenRequest.builder()
            .document_id(doc_token)
            .block_id(doc_token)
            .request_body(body)
            .build()
        )
        insert_resp = await _lark_thread(client.docx.v1.document_block_children.create, req)
        if insert_resp.success() and insert_resp.data and insert_resp.data.blocks:
            return f"Image uploaded and inserted: {file_name} (block_id={insert_resp.data.blocks[0].block_id})"
        return f"[error] Image block insertion failed: {insert_resp.code} {insert_resp.msg}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"
