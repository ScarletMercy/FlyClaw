"""Embedding provider using OpenAI-compatible API."""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from src.config import MemoryConfig, ModelConfig

logger = logging.getLogger("myclaw.memory.embeddings")


class EmbeddingProvider:
    """Embedding provider using OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(self, config: MemoryConfig, model_config: ModelConfig):
        self.config = config
        self._model = config.embedding_model
        self._dimensions = config.embedding_dimensions
        api_key = config.api_key or model_config.api_key or ""
        base_url = (config.base_url or model_config.base_url or "https://api.openai.com").rstrip("/")
        self._url = f"{base_url}/v1/embeddings"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch embed texts. Returns list of float vectors."""
        if not texts:
            return []

        # OpenAI supports batch embedding; process in groups of 100
        all_embeddings: list[list[float]] = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {
                "model": self._model,
                "input": batch,
            }
            if self._dimensions:
                payload["dimensions"] = self._dimensions

            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(self._url, json=payload, headers=self._headers)
                    resp.raise_for_status()
                    data = resp.json()
                    # Sort by index to ensure correct order
                    items = sorted(data["data"], key=lambda x: x["index"])
                    all_embeddings.extend([item["embedding"] for item in items])
            except Exception as e:
                logger.error("Embedding batch failed (texts %d-%d): %s", i, i + len(batch), e)
                # Fill with zero vectors on failure
                for _ in batch:
                    all_embeddings.append([0.0] * self._dimensions)

        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        results = await self.embed_texts([text])
        return results[0] if results else []
