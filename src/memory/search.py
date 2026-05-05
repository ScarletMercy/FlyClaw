"""High-level memory search combining embedding and store."""

from __future__ import annotations

import logging
from typing import Optional

from src.config import MemoryConfig
from src.memory.base import BaseMemoryStore
from src.memory.embeddings import EmbeddingProvider

logger = logging.getLogger("myclaw.memory.search")


class MemorySearcher:
    """End-to-end memory search: embed query → hybrid search → format results."""

    def __init__(self, store: BaseMemoryStore, embeddings: EmbeddingProvider, config: MemoryConfig):
        self.store = store
        self.embeddings = embeddings
        self.config = config

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> list[dict]:
        """Search memory for relevant chunks.

        Returns list of dicts with keys: content, path, score, chunk_index.
        """
        max_results = max_results or self.config.max_results

        # Get query embedding for vector search
        query_embedding = None
        if self.store.dimensions > 0:
            try:
                query_embedding = await self.embeddings.embed_query(query)
            except Exception as e:
                logger.warning("Query embedding failed, using FTS5-only: %s", e)

        # Hybrid search
        results = await self.store.search(
            query_embedding=query_embedding,
            query_text=query,
            max_results=max_results,
            vector_weight=self.config.vector_weight,
            min_score=self.config.min_score,
        )

        # Format for agent consumption
        formatted = []
        for r in results:
            formatted.append(
                {
                    "content": r["content"],
                    "path": r["path"],
                    "score": round(r.get("score", 0), 3),
                    "chunk_index": r.get("chunk_index", 0),
                }
            )
        return formatted

    async def index_document(self, path: str, content: str) -> int:
        """Index a document: chunk, embed, and store.

        Returns number of chunks added.
        """
        # Add chunks to store
        count = await self.store.add_document(path, content)
        if count == 0:
            return 0

        # Get chunk IDs and embed
        chunk_ids = await self.store.get_chunk_ids_for_path(path)
        if not chunk_ids:
            return count

        try:
            # Re-chunk to get text for embedding
            from src.memory.chunker import chunk_markdown

            chunks = chunk_markdown(content)
            texts = [c["text"] for c in chunks[: len(chunk_ids)]]
            embeddings = await self.embeddings.embed_texts(texts)
            await self.store.add_embeddings(chunk_ids[: len(embeddings)], embeddings)
        except Exception as e:
            logger.warning("Embedding for document %s failed: %s", path, e)

        return count
