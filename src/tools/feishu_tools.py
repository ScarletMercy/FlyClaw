from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger("myclaw.feishu_tools")


def _get_feishu_client():
    from src.channels.feishu import get_feishu_client

    return get_feishu_client()


@tool
async def feishu_get_doc_content(doc_id: str) -> str:
    """Read the content of a Feishu document.

    Args:
        doc_id: The document ID or URL.
    """
    client = _get_feishu_client()
    if client is None:
        return "[error] Feishu client not initialized"
    try:
        import lark_oapi as lark
        from lark_oapi.api.docx_v1 import (
            GetDocumentRawContentRequest,
            GetDocumentRawContentRequestBuilder,
        )

        req = GetDocumentRawContentRequestBuilder().request(GetDocumentRawContentRequest()).build()
        resp = await asyncio.to_thread(
            client.docx.v1.document.raw_content.get,
            req,
            doc_id,
        )
        if resp.success():
            return resp.data.content or "(empty document)"
        logger.error("Feishu get_doc_content failed for doc_id=%s: %s: %s", doc_id, resp.code, resp.msg)
        return f"[error] {resp.code}: {resp.msg}"
    except Exception as e:
        logger.error("Feishu get_doc_content error for doc_id=%s: %s", doc_id, e)
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
        import lark_oapi as lark
        from lark_oapi.api.im_v1 import GetChatRequest

        req = GetChatRequest.builder().chat_id(chat_id).build()
        resp = await asyncio.to_thread(client.im.v1.chat.get, req)
        if resp.success() and resp.data:
            info = {
                "name": resp.data.name,
                "chat_id": resp.data.chat_id,
                "owner_id": resp.data.owner_id,
                "member_count": resp.data.user_count,
            }
            return json.dumps(info, ensure_ascii=False)
        logger.error("Feishu get_chat_info failed for chat_id=%s: %s: %s", chat_id, resp.code, resp.msg)
        return f"[error] {resp.code}: {resp.msg}"
    except Exception as e:
        logger.error("Feishu get_chat_info error for chat_id=%s: %s", chat_id, e)
        return f"[error] {type(e).__name__}: {e}"


import asyncio
