"""Embedding provider using OpenAI-compatible API.

走 openai SDK（与 ChatClient 一致），base_url 透传给 SDK 拼 URL。
SDK 期望 base_url 已含版本段（如 `…/v1`），追加 `/embeddings`；
因此**不要**自己再拼 `/v1/embeddings`——历史上那样做会在 base_url 已带
`/v1` 时产生 `…/v1/v1/embeddings` → 404（DeepSeek/Groq/智谱/DashScope 全中）。
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from openai import AsyncOpenAI

from src.config import MemoryConfig, MemoryStoreConfig, ModelConfig

logger = logging.getLogger("flyclaw.memory.embeddings")


class EmbeddingProvider:
    """Embedding provider using the openai SDK's /embeddings endpoint."""

    def __init__(self, config: MemoryConfig, model_config: ModelConfig):
        self.config = config
        self._model = getattr(config, "embedding_model", "text-embedding-3-small")
        self._dimensions = getattr(config, "embedding_dimensions", 1536)
        api_key = getattr(config, "api_key", "") or model_config.api_key or ""
        base_url = (getattr(config, "base_url", "") or model_config.base_url or "").rstrip("/")
        # base_url 留空时 _get_client 传 None → SDK 走默认 https://api.openai.com/v1
        self._api_key = api_key
        self._base_url = base_url
        self._client: Optional[AsyncOpenAI] = None  # 懒构造：空 api_key 时不在构造期抛

    @classmethod
    def from_vector_config(cls, ms: MemoryStoreConfig, model_config: ModelConfig) -> "EmbeddingProvider":
        """从 MemoryStoreConfig.vector_* 字段构造，不 fallback 到 model_config。

        用 __new__ 绕过 __init__ 的 getattr fallback 逻辑，确保 vector_* 留空时
        不会泄漏 model_config.api_key / base_url。客户端懒构造，空凭据在
        embed_texts 实际调用时才抛 Missing credentials（而非构造期）。
        """
        obj = cls.__new__(cls)
        obj.config = ms
        obj._model = ms.vector_model or "text-embedding-3-small"
        obj._dimensions = ms.vector_dimensions or 1536
        obj._api_key = ms.vector_api_key or ""  # 不 fallback 到 model_config
        obj._base_url = (ms.vector_base_url or "").rstrip("/")
        obj._client = None
        return obj

    def _get_client(self) -> AsyncOpenAI:
        """懒构造 SDK 客户端。base_url 留空 → None → SDK 默认；不自己追加 /v1。"""
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self._base_url or None,
                api_key=self._api_key or None,
                timeout=httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0),
                max_retries=2,
            )
        return self._client

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch embed texts. Returns list of float vectors.

        Raises on any batch failure to prevent index misalignment
        (partial failures would silently shift subsequent embeddings
        to wrong indices). Callers should handle exceptions.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        batch_size = 10  # doubao-embedding-vision 等限制单次 input ≤10；OpenAI 无副作用
        client = self._get_client()

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # 按 OpenAI 规范只发 model + input；dimensions 是可选参数，多发反而让不支持
            # 短维度的兼容模型 400。用模型原生维度，_dimensions 由向导探测，仅用于 schema + 一致性校验。
            resp = await client.embeddings.create(model=self._model, input=batch)
            # Sort by index to ensure correct order
            items = sorted(resp.data, key=lambda x: x.index)
            batch_embeddings = [list(item.embedding) for item in items]
            # 维度校验：provider 忽略 dimensions 参数时返回原生维度，下游 pa.list_ 会抛
            # （add_document 已 commit → 孤儿 chunk）。这里早抛让调用方降级（migration→FTS5-only）
            if self._dimensions:
                for j, emb in enumerate(batch_embeddings):
                    if len(emb) != self._dimensions:
                        raise RuntimeError(
                            f"Embedding dim mismatch: model returned {len(emb)}, "
                            f"expected {self._dimensions} (item {j}). "
                            f"Provider may ignore the dimensions param."
                        )
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string. Raises RuntimeError on failure."""
        results = await self.embed_texts([text])
        if not results:
            raise RuntimeError("Embedding query failed — no results returned")
        return results[0]

    async def close(self) -> None:
        """Close the underlying SDK HTTP connection pool."""
        if self._client is not None:
            await self._client.close()
            self._client = None
