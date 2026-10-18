"""Shared FTS5 tokenization & query sanitization utilities."""

from __future__ import annotations

import re

# CJK 连续段：unicode61 分词器不做中文分词，整段汉字会成一个 token，
# 必须由应用层预切——单字成 token，配合查询侧短语匹配等价子串查找
_CJK_RUN = re.compile(r"[一-鿿]+")
_ASCII_TOKEN = re.compile(r"[a-zA-Z0-9_]+")


def tokenize_for_fts(text: str | None) -> str:
    """索引侧切分：CJK 段逐字插空格（单字即 token），其余文本原样保留。

    查询侧（sanitize_fts5_query）必须使用同一套切法——索引与查询的
    token 粒度一致，MATCH 才能命中。
    """
    if not text:
        return ""
    parts: list[str] = []
    pos = 0
    for m in _CJK_RUN.finditer(text):
        if m.start() > pos:
            parts.append(text[pos : m.start()])
        parts.append(" ".join(m.group()))
        pos = m.end()
    if pos < len(text):
        parts.append(text[pos:])
    return " ".join(parts)


def tokenize_batch(texts: list[str | None]) -> list[str]:
    return [tokenize_for_fts(t) for t in texts]


def sanitize_fts5_query(query: str) -> str:
    """Build a safe FTS5 MATCH query from raw user input.

    - CJK 段按字切、以短语（相邻 token）匹配——等价子串查找；长段
      降级为双字短语 OR，避免整句连续匹配漏召回。
    - ASCII 提取字母数字 token；段与段之间 AND。
    - Strips FTS5-special characters (``"`` ``*`` ``(`` ``)``) before
      tokenization；引号内均为裸 token 字符，无语法注入面。
    - Returns ``'""'`` for empty / whitespace-only input.
    """
    query = query.strip()
    if not query:
        return '""'
    # Strip FTS5-special characters so they cannot interfere with tokenization
    query = query.replace('"', " ").replace("*", " ").replace("(", " ").replace(")", " ")

    terms: list[str] = []
    for seg in _CJK_RUN.findall(query):
        if len(seg) <= 4:
            terms.append(f'"{" ".join(seg)}"')
        else:
            bigrams = (f'"{" ".join(seg[i : i + 2])}"' for i in range(len(seg) - 1))
            terms.append("(" + " OR ".join(bigrams) + ")")
    terms.extend(f'"{w}"' for w in _ASCII_TOKEN.findall(query))

    return " AND ".join(terms) if terms else '""'
