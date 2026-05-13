"""LLM-based semantic search over session history."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger("myclaw.session_index.search")

SYSTEM_PROMPT = """你是一个会话搜索助手。用户给你一个搜索意图，你从候选会话列表中找出相关的会话。

规则：
1. 返回 JSON 数组，包含相关会话的 index（从0开始）
2. 只返回真正相关的会话，不相关返回空数组 []
3. 每个会话附一句简短理由（20字以内）
4. 最多返回 5 个会话

返回格式（纯JSON，不要markdown）：
[{"index": 0, "reason": "讨论了Docker部署"}, {"index": 2, "reason": "包含相关代码"}]

如果没有任何相关会话，返回：[]"""


async def llm_search(
    store,
    query: str,
    model_name: str,
    base_url: str,
    api_key: str,
    max_candidates: int = 20,
    max_results: int = 5,
) -> list[dict]:
    """Use a small model to semantically search sessions.

    1. Get candidate sessions from index (recent + FTS5 broad match)
    2. Send to model for semantic filtering
    3. Return matched sessions with reasons
    """
    # 1. Gather candidates: recent sessions + FTS5 results
    candidates = _gather_candidates(store, query, max_candidates)
    if not candidates:
        return []

    # 2. Build prompt with session summaries
    candidate_text = _format_candidates(candidates)

    # 3. Call model
    try:
        response_text = await _call_model(
            model_name, base_url, api_key, query, candidate_text,
        )
    except Exception as e:
        logger.warning("LLM search failed, returning raw candidates: %s", e)
        # Fallback: return candidates without LLM filtering
        return _candidates_to_results(candidates[:max_results])

    # 4. Parse model response
    return _parse_llm_response(response_text, candidates, max_results)


def _gather_candidates(store, query: str, max_candidates: int) -> list[dict]:
    """Get candidate sessions from recent list + FTS5 results, deduplicated."""
    seen_thread_ids = set()
    candidates = []

    # First: FTS5 results (if query is non-empty)
    if query.strip():
        fts_results = store.search(query, limit=max_candidates)
        for r in fts_results:
            if r["thread_id"] not in seen_thread_ids:
                seen_thread_ids.add(r["thread_id"])
                candidates.append(r)

    # Then: recent sessions to fill up to max_candidates
    if len(candidates) < max_candidates:
        recent = store.search("", limit=max_candidates)
        for r in recent:
            if r["thread_id"] not in seen_thread_ids:
                seen_thread_ids.add(r["thread_id"])
                candidates.append(r)

    return candidates[:max_candidates]


def _format_candidates(candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates):
        snippet = (c.get("snippet") or "").replace("\n", " ")[:150]
        msg_count = c.get("message_count", 0)
        channel = c.get("channel", "?")
        lines.append(f"[{i}] ({channel}, {msg_count}条消息) {snippet}")
    return "\n".join(lines)


async def _call_model(
    model_name: str,
    base_url: str,
    api_key: str,
    query: str,
    candidate_text: str,
) -> str:
    """Call OpenAI-compatible chat completion API."""
    import httpx

    user_msg = f"搜索意图：{query}\n\n候选会话：\n{candidate_text}"

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        resp = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.0,
                "max_tokens": 500,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def _parse_llm_response(
    response_text: str,
    candidates: list[dict],
    max_results: int,
) -> list[dict]:
    """Parse LLM JSON response into search results."""
    # Strip markdown code fences if present
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        selections = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("LLM search returned invalid JSON: %.200s", response_text)
        return _candidates_to_results(candidates[:max_results])

    if not isinstance(selections, list):
        return _candidates_to_results(candidates[:max_results])

    results = []
    for sel in selections[:max_results]:
        idx = sel.get("index")
        if isinstance(idx, int) and 0 <= idx < len(candidates):
            r = dict(candidates[idx])
            r["reason"] = sel.get("reason", "")
            results.append(r)

    return results


def _candidates_to_results(candidates: list[dict]) -> list[dict]:
    """Convert raw candidates to result format."""
    results = []
    for c in candidates:
        r = dict(c)
        r["reason"] = ""
        results.append(r)
    return results
