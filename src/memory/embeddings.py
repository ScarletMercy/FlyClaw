"""Embedding provider using OpenAI-compatible API."""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from src.config import MemoryConfig, ModelConfig

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
                all_embeddings.extend([item["embedding"] for item in items])

        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string. Raises RuntimeError on failure."""
        results = await self.embed_texts([text])
        if not results:
            raise RuntimeError("Embedding query failed — no results returned")
        return results[0]
