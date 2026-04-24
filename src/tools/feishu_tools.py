from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger("myclaw.feishu_tools")


def _get_feishu_client():
    from src.channels.feishu import get_feishu_client
    return get_feishu_client()


@tool
async def feishu_send_message(chat_id: str, text: str, msg_type: str = "text") -> str:
    """Send a text or interactive message to a Feishu chat.

    Args:
        chat_id: The Feishu chat ID (e.g. "oc_xxx").
        text: Message text content. For text type, plain text. For interactive, JSON card content.
        msg_type: "text" (default) or "interactive" (card JSON).
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        if msg_type == "interactive":
            content = text
        else:
            content = json.dumps({"text": text}, ensure_ascii=False)

        body = (
            CreateMessageRequestBody.builder()
            .receive_id(chat_id)
            .msg_type(msg_type)
            .content(content)
            .build()
        )
        req = CreateMessageRequest.builder().receive_id_type("chat_id").request_body(body).build()
        resp = await asyncio.to_thread(client.im.v1.message.create, req)
        if resp.success() and resp.data:
            return f"Message sent to {chat_id}, message_id={resp.data.message_id}"
        return f"[error] {resp.code}: {resp.msg}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


@tool
async def feishu_get_user_info(user_id: str) -> str:
    """Get information about a Feishu user by open_id or user_id.

    Args:
        user_id: User's open_id, user_id, or union_id.
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        from lark_oapi.api.contact.v3 import GetUserRequest, UserIdType

        req = GetUserRequest.builder().user_id(user_id).user_id_type("open_id").build()
        resp = await asyncio.to_thread(client.contact.v3.user.get, req)
        if resp.success() and resp.data.user:
            u = resp.data.user
            info = {
                "name": u.name,
                "open_id": u.open_id,
                "union_id": u.union_id,
                "user_id": u.user_id,
                "email": u.email,
                "mobile": u.mobile,
                "department_ids": u.department_ids,
            }
            return json.dumps(info, ensure_ascii=False, default=str)
        return f"[error] {resp.code}: {resp.msg}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


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
            resp = await asyncio.to_thread(client.im.v1.chat_member.get, req)
            if not resp.success():
                return f"[error] {resp.code}: {resp.msg}"
            if resp.data and resp.data.items:
                for item in resp.data.items:
                    members.append({
                        "member_id": item.member_id,
                        "member_id_type": item.member_id_type,
                        "name": getattr(item, "name", ""),
                    })
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
        resp = await asyncio.to_thread(client.im.v1.chat.create, req)
        if not resp.success():
            return f"[error] {resp.code}: {resp.msg}"
        chat_id = resp.data.chat_id if resp.data else ""
        result = f"Chat created: {chat_id} (name={name})"

        if user_ids and chat_id:
            uid_list = [uid.strip() for uid in user_ids.split(",") if uid.strip()]
            for uid in uid_list[:50]:
                from lark_oapi.api.im.v1 import AddChatMemberRequest, AddChatMemberRequestBody
                member_body = AddChatMemberRequestBody.builder().member_list(
                    json.dumps([{"member_id": uid, "member_id_type": "open_id"}])
                ).build()
                member_req = AddChatMemberRequest.builder().chat_id(chat_id).request_body(member_body).build()
                mr = await asyncio.to_thread(client.im.v1.chat_member.create, member_req)
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
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        from lark_oapi.api.im.v1 import GetChatRequest

        req = GetChatRequest.builder().chat_id(chat_id).build()
        resp = await asyncio.to_thread(client.im.v1.chat.get, req)
        if resp.success() and resp.data:
            info = {
                "name": resp.data.name,
                "chat_id": resp.data.chat_id,
                "owner_id": resp.data.owner_id,
                "member_count": resp.data.user_count,
                "description": getattr(resp.data, "description", ""),
                "chat_type": getattr(resp.data, "chat_type", ""),
            }
            return json.dumps(info, ensure_ascii=False, default=str)
        return f"[error] {resp.code}: {resp.msg}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


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
            from lark_oapi.api.wiki_v2 import GetNodeRequest

            wiki_req = GetNodeRequest.builder().token(doc_token).build()
            wiki_resp = await asyncio.to_thread(client.wiki.v2.space_node.get, wiki_req)
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
async def feishu_list_calendar_events(start_time: str, end_time: str, user_id: str = "") -> str:
    """List Feishu calendar events in a time range.

    Args:
        start_time: Start time in ISO format (e.g. "2026-04-18T00:00:00+08:00").
        end_time: End time in ISO format.
        user_id: User's open_id (empty for bot's calendar).
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        from datetime import datetime as _dt

        from lark_oapi.api.calendar.v4 import ListCalendarEventRequest

        start_ts = int(_dt.fromisoformat(start_time).timestamp() * 1000)
        end_ts = int(_dt.fromisoformat(end_time).timestamp() * 1000)
        cal_id = f"user_{user_id}" if user_id else ""

        req = (
            ListCalendarEventRequest.builder()
            .calendar_id(cal_id)
            .start_time(start_ts)
            .end_time(end_ts)
            .page_size(50)
            .build()
        )
        resp = await asyncio.to_thread(client.calendar.v4.calendar_event.list, req)
        if resp.success() and resp.data and resp.data.items:
            events = []
            for ev in resp.data.items:
                events.append({
                    "summary": getattr(ev, "summary", ""),
                    "start_time": str(getattr(ev, "start_time", "")),
                    "end_time": str(getattr(ev, "end_time", "")),
                    "description": getattr(ev, "description", "")[:200],
                })
            return json.dumps(events, ensure_ascii=False, default=str)
        return "No calendar events found"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


@tool
async def feishu_create_document(title: str, folder_token: str = "") -> str:
    """Create a new Feishu cloud document (docx) and return its URL.

    Args:
        title: Document title.
        folder_token: Target folder token (empty for root folder).
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        from lark_oapi.api.docx.v1 import CreateDocumentRequest, CreateDocumentRequestBody

        body = CreateDocumentRequestBody.builder().title(title).folder_token(folder_token or "").build()
        req = CreateDocumentRequest.builder().request_body(body).build()
        resp = await asyncio.to_thread(client.docx.v1.document.create, req)
        if resp.success() and resp.data:
            doc = resp.data.document
            url = getattr(doc, "url", "") if doc else ""
            token = getattr(doc, "document_id", "") if doc else ""
            result = f"Document created: {title}"
            if url:
                result += f"\nURL: {url}"
            if token:
                result += f"\nToken: {token}"
            return result
        return f"[error] {resp.code}: {resp.msg}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


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
    import os

    abs_path = os.path.abspath(file_path)
    if not os.path.isfile(abs_path):
        return f"[error] File not found: {abs_path}"
    try:
        from lark_oapi.api.drive.v1 import UploadAllMediaRequest, UploadAllMediaRequestBody

        file_name = os.path.basename(abs_path)
        file_size = os.path.getsize(abs_path)

        def _do_upload():
            with open(abs_path, "rb") as f:
                body = (
                    UploadAllMediaRequestBody.builder()
                    .file_name(file_name)
                    .parent_type("drive")
                    .parent_node(folder_token or "0")
                    .size(file_size)
                    .file(f)
                    .build()
                )
                req = UploadAllMediaRequest.builder().request_body(body).build()
                return client.drive.v1.media.upload_all(req)

        resp = await asyncio.to_thread(_do_upload)
        if resp.success() and resp.data:
            return f"File uploaded: {file_name} (token: {resp.data.file_token})"
        return f"[error] Upload failed: {resp.code}: {resp.msg}"
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
            GetDocumentBlockChildrenRequest.builder()
            .document_id(doc_token)
            .block_id(doc_token)
            .page_size(500)
            .build()
        )
        get_resp = await asyncio.to_thread(
            client.docx.v1.document_block_children.get, get_req
        )
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
            del_resp = await asyncio.to_thread(
                client.docx.v1.document_block_children.batch_delete, del_req
            )
            if del_resp.success():
                deleted = children_count
            else:
                logger.warning("Failed to clear document content: %s %s", del_resp.code, del_resp.msg)

        # 3. Convert markdown to Feishu blocks via SDK
        convert_body = (
            ConvertDocumentRequestBody.builder()
            .content_type("markdown")
            .content(content)
            .build()
        )
        convert_req = (
            ConvertDocumentRequest.builder()
            .request_body(convert_body)
            .build()
        )
        convert_resp = await asyncio.to_thread(
            client.docx.v1.document.convert, convert_req
        )
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
        insert_resp = await asyncio.to_thread(
            client.docx.v1.document_block_descendant.create, insert_req
        )
        if insert_resp.success():
            return f"Document updated: {doc_token} ({deleted} cleared, {len(blocks)} blocks inserted)"
        return f"[error] Block insertion failed: {insert_resp.code} {insert_resp.msg}"
    except Exception as e:
        logger.error("feishu_write_document error: %s", e, exc_info=True)
        return f"[error] {type(e).__name__}: {e}"


@tool
async def feishu_create_calendar_event(
    calendar_id: str,
    title: str,
    start_time: str,
    end_time: str,
    description: str = "",
) -> str:
    """Create a Feishu calendar event.

    Args:
        calendar_id: Calendar ID to create the event in. Use feishu_list_calendar_events to find it.
        title: Event title.
        start_time: Start time in ISO format (e.g. "2026-04-18T14:00:00+08:00").
        end_time: End time in ISO format.
        description: Event description.
    """
    try:
        from src.channels.feishu import _resolve_api_base, _get_tenant_token, feishu_api_request
        from src.config import load_config

        cfg = load_config()
        domain = cfg.channels.feishu.domain
        token = await _get_tenant_token(cfg.channels.feishu.app_id, cfg.channels.feishu.app_secret, domain)
        if not token:
            return "[error] Failed to get tenant token"

        api_base = _resolve_api_base(domain)
        from datetime import datetime

        start_ts = int(datetime.fromisoformat(start_time).timestamp())
        end_ts = int(datetime.fromisoformat(end_time).timestamp())

        payload = {
            "summary": title,
            "description": description,
            "start_time": {"timestamp": str(start_ts)},
            "end_time": {"timestamp": str(end_ts)},
        }

        import httpx
        async with httpx.AsyncClient(timeout=30) as hc:
            resp = await feishu_api_request(
                lambda: hc.post(
                    f"{api_base}/open-apis/calendar/v4/calendars/{calendar_id}/events",
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json=payload,
                ),
                description="Create calendar event",
            )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == 0:
                return f"Calendar event created: {title}"
            return f"[error] {data.get('code')}: {data.get('msg')}"
        return f"[error] HTTP {resp.status_code}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


@tool
async def feishu_get_message_list(chat_id: str, count: int = 20) -> str:
    """Get recent messages from a Feishu chat.

    Args:
        chat_id: The chat ID.
        count: Number of messages to retrieve (max 50).
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        from lark_oapi.api.im.v1 import ListMessageRequest

        req = ListMessageRequest.builder().container_id_type("chat").container_id(chat_id).page_size(min(count, 50)).build()
        resp = await asyncio.to_thread(client.im.v1.message.list, req)
        if resp.success() and resp.data and resp.data.items:
            messages = []
            for item in resp.data.items:
                msg = item.message if hasattr(item, 'message') else None
                if not msg:
                    continue
                body_content = getattr(msg.body, 'content', '') if msg.body else ''
                text = ""
                try:
                    body = json.loads(body_content) if body_content else {}
                    text = body.get("text", body.get("content", str(body)))
                except Exception:
                    text = body_content
                sender = getattr(msg, 'sender', None)
                sender_id = ""
                if sender and hasattr(sender, 'sender_id') and sender.sender_id:
                    sender_id = sender.sender_id.open_id
                messages.append(f"[{msg.message_type}] {sender_id}: {text[:200]}")
            return "\n---\n".join(messages) if messages else "No readable messages found"
        return "No messages found"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


@tool
async def feishu_recall_message(message_id: str) -> str:
    """Recall (delete) a previously sent message.

    Args:
        message_id: The message ID to recall.
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        from lark_oapi.api.im.v1 import DeleteMessageRequest

        req = DeleteMessageRequest.builder().message_id(message_id).build()
        resp = await asyncio.to_thread(client.im.v1.message.delete, req)
        if resp.success():
            return f"Message recalled: {message_id}"
        return f"[error] {resp.code}: {resp.msg}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


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
                api_token = await _get_tenant_token(
                    cfg.channels.feishu.app_id, cfg.channels.feishu.app_secret, domain
                )
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
        body = (
            CreateFolderFileRequestBody.builder()
            .name(name)
            .folder_token(effective_token)
            .build()
        )
        req = CreateFolderFileRequest.builder().request_body(body).build()
        resp = await asyncio.to_thread(client.drive.v1.file.create_folder, req)

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
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        from lark_oapi.api.drive.v1 import ListFileRequest

        req = ListFileRequest.builder().folder_token(folder_token or "").build()
        resp = await asyncio.to_thread(client.drive.v1.file.list, req)
        if resp.success() and resp.data and resp.data.files:
            items = []
            for f in resp.data.files:
                items.append({
                    "name": f.name,
                    "token": f.token,
                    "type": f.type,
                })
            return json.dumps(items, ensure_ascii=False, default=str)
        return "No files found"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


@tool
async def feishu_send_card(chat_id: str, title: str, content: str) -> str:
    """Send an interactive card message to a Feishu chat.

    Args:
        chat_id: The Feishu chat ID.
        title: Card title.
        content: Card body text (supports markdown).
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        card = {
            "schema": "2.0",
            "header": {"title": {"tag": "plain_text", "content": title}},
            "body": {
                "elements": [
                    {"tag": "markdown", "content": content}
                ]
            },
        }
        msg_content = json.dumps(card, ensure_ascii=False)
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        body = (
            CreateMessageRequestBody.builder()
            .receive_id(chat_id)
            .msg_type("interactive")
            .content(msg_content)
            .build()
        )
        req = CreateMessageRequest.builder().receive_id_type("chat_id").request_body(body).build()
        resp = await asyncio.to_thread(client.im.v1.message.create, req)
        if resp.success() and resp.data:
            return f"Card sent to {chat_id}, message_id={resp.data.message_id}"
        return f"[error] {resp.code}: {resp.msg}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


@tool
async def feishu_create_bitable(name: str, folder_token: str = "") -> str:
    """Create a new Feishu bitable (multi-dimensional table).

    Args:
        name: Bitable name.
        folder_token: Parent folder token (empty for root).
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        from lark_oapi.api.bitable.v1 import CreateAppRequest

        body = json.dumps({"name": name, "folder_token": folder_token})
        req = CreateAppRequest.builder().request_body(body).build()
        resp = await asyncio.to_thread(client.bitable.v1.app.create, req)
        if resp.success() and resp.data:
            app = resp.data.app
            return f"Bitable created: {app.name} (app_token={app.app_token})"
        return f"[error] {resp.code}: {resp.msg}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


@tool
async def feishu_bitable_list_records(app_token: str, table_id: str, page_size: int = 20) -> str:
    """List records from a Feishu bitable table.

    Args:
        app_token: Bitable app token.
        table_id: Table ID within the bitable.
        page_size: Max records to fetch (max 500).
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        from lark_oapi.api.bitable.v1 import ListAppTableRecordRequest

        req = ListAppTableRecordRequest.builder().app_token(app_token).table_id(table_id).page_size(min(page_size, 500)).build()
        resp = await asyncio.to_thread(client.bitable.v1.app_table_record.list, req)
        if resp.success() and resp.data and resp.data.items:
            records = []
            for r in resp.data.items:
                records.append(r.fields if hasattr(r, 'fields') else str(r))
            return json.dumps(records, ensure_ascii=False, default=str)
        return "No records found"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


@tool
async def feishu_bitable_add_record(app_token: str, table_id: str, fields_json: str) -> str:
    """Add a record to a Feishu bitable table.

    Args:
        app_token: Bitable app token.
        table_id: Table ID.
        fields_json: Record fields as JSON object, e.g. '{"Name": "test", "Age": 25}'.
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        from lark_oapi.api.bitable.v1 import CreateAppTableRecordRequest

        fields = json.loads(fields_json)
        body = json.dumps({"fields": fields})
        req = CreateAppTableRecordRequest.builder().app_token(app_token).table_id(table_id).request_body(body).build()
        resp = await asyncio.to_thread(client.bitable.v1.app_table_record.create, req)
        if resp.success() and resp.data:
            record = resp.data.record
            return f"Record added (record_id={record.record_id})"
        return f"[error] {resp.code}: {resp.msg}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"
