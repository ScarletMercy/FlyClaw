"""Embedding provider using OpenAI-compatible API."""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from src.config import MemoryConfig, MemoryStoreConfig, ModelConfig

logger = logging.getLogger("flyclaw.memory.embeddings")


class EmbeddingProvider:
    """Embedding provider using OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(self, config: MemoryConfig, model_config: ModelConfig):
        self.config = config
        self._model = getattr(config, "embedding_model", "text-embedding-3-small")
        self._dimensions = getattr(config, "embedding_dimensions", 1536)
        api_key = getattr(config, "api_key", "") or model_config.api_key or ""
        base_url = (getattr(config, "base_url", "") or model_config.base_url or "https://api.openai.com").rstrip("/")
        self._url = f"{base_url}/v1/embeddings"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @classmethod
    def from_vector_config(cls, ms: MemoryStoreConfig, model_config: ModelConfig) -> "EmbeddingProvider":
        """从 MemoryStoreConfig.vector_* 字段构造，不 fallback 到 model_config。

        用 __new__ 绕过 __init__ 的 getattr fallback 逻辑，确保 vector_* 留空时
        不会泄漏 model_config.api_key / base_url。
        """
        obj = cls.__new__(cls)
        obj.config = ms
        obj._model = ms.vector_model or "text-embedding-3-small"
        obj._dimensions = ms.vector_dimensions or 1536
        base_url = (ms.vector_base_url or "").rstrip("/")
        obj._url = f"{base_url}/v1/embeddings"
        obj._headers = {
            "Authorization": f"Bearer {ms.vector_api_key}",
            "Content-Type": "application/json",
        }
        return obj

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch embed texts. Returns list of float vectors.

        Raises on any batch failure to prevent index misalignment
        (partial failures would silently shift subsequent embeddings
        to wrong indices). Callers should handle exceptions.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        batch_size = 100

        async with httpx.AsyncClient(timeout=60.0) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                payload = {
                    "model": self._model,
                    "input": batch,
                }
                if self._dimensions:
                    payload["dimensions"] = self._dimensions

                resp = await client.post(self._url, json=payload, headers=self._headers)
                resp.raise_for_status()
                data = resp.json()
                # Sort by index to ensure correct order
                items = sorted(data["data"], key=lambda x: x["index"])
                batch_embeddings = [item["embedding"] for item in items]
                # 维度校验：provider 忽略 dimensions 参数时返回原生维度，下游 pa.list_ 会抛
                # （add_document 已 commit → 孤儿 chunk）。这里早抛让调用方降级（migration→FTS5-only）
                if self._dimensions:
                    for i, emb in enumerate(batch_embeddings):
                        if len(emb) != self._dimensions:
                            raise RuntimeError(
                                f"Embedding dim mismatch: model returned {len(emb)}, "
                                f"expected {self._dimensions} (item {i}). "
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
