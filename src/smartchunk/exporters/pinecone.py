"""Pinecone vector database exporter.

Requires the ``pinecone`` extra::

    pip install smartchunk[pinecone]
"""

from __future__ import annotations

import logging
import os
from typing import Any

from smartchunk.exporters.base import BaseExporter
from smartchunk.models import SmartChunk

logger = logging.getLogger("smartchunk.exporters.pinecone")


class PineconeExporter(BaseExporter):
    """Exports SmartChunks to a Pinecone index.

    Embeds chunk text using a configurable embedding model and upserts
    vectors with all SmartChunk metadata as Pinecone metadata fields.
    """

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        batch_size: int = 100,
    ) -> None:
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY", "")
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.batch_size = batch_size

    def export(self, chunks: list[SmartChunk], **kwargs: object) -> None:
        """Export chunks to Pinecone.

        Parameters
        ----------
        chunks:
            List of enriched SmartChunks.
        **kwargs:
            Optional overrides: ``index_name``, ``namespace``.
        """
        index_name = str(kwargs.get("index_name", self.index_name or "smartchunk"))
        namespace = str(kwargs.get("namespace", ""))

        self.to_pinecone(
            chunks,
            index_name=index_name,
            api_key=self.api_key,
            namespace=namespace,
        )

    def to_pinecone(
        self,
        chunks: list[SmartChunk],
        index_name: str,
        api_key: str | None = None,
        namespace: str = "",
    ) -> None:
        """Upload chunks to Pinecone with embeddings and metadata.

        Parameters
        ----------
        chunks:
            List of SmartChunks to upload.
        index_name:
            Name of the Pinecone index.
        api_key:
            Pinecone API key. Falls back to env var PINECONE_API_KEY.
        namespace:
            Optional Pinecone namespace.
        """
        try:
            from pinecone import Pinecone  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Pinecone export requires pinecone-client. "
                "Install with: pip install smartchunk[pinecone]"
            ) from exc

        from litellm import embedding as litellm_embedding  # type: ignore[import-untyped]

        api_key = api_key or self.api_key
        if not api_key:
            raise ValueError("Pinecone API key required. Set PINECONE_API_KEY or pass api_key.")

        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)

        # Process in batches
        for batch_start in range(0, len(chunks), self.batch_size):
            batch = chunks[batch_start : batch_start + self.batch_size]

            # Get embeddings using contextual_text
            texts = [chunk.contextual_text or chunk.text for chunk in batch]
            response = litellm_embedding(model=self.embedding_model, input=texts)
            embeddings = [item["embedding"] for item in response.data]

            # Build upsert vectors
            vectors: list[dict[str, Any]] = []
            for chunk, emb in zip(batch, embeddings):
                metadata = {
                    "text": chunk.text,
                    "contextual_text": chunk.contextual_text,
                    "summary": chunk.summary,
                    "entities": chunk.entities,
                    "keywords": chunk.keywords,
                    "parent_context": chunk.parent_context,
                    "prev_summary": chunk.prev_summary,
                    "next_summary": chunk.next_summary,
                    "prev_id": chunk.prev_id or "",
                    "next_id": chunk.next_id or "",
                    "parent_id": chunk.parent_id or "",
                    "section_id": chunk.section_id or "",
                    "depth": chunk.depth,
                    "confidence": chunk.confidence,
                    "source": chunk.metadata.source,
                    "chunk_index": chunk.metadata.chunk_index,
                    "total_chunks": chunk.metadata.total_chunks,
                    "page": chunk.metadata.page or 0,
                }
                vectors.append(
                    {
                        "id": chunk.id,
                        "values": emb,
                        "metadata": metadata,
                    }
                )

            index.upsert(vectors=vectors, namespace=namespace)
            logger.info(
                "Upserted %d vectors to Pinecone index '%s'",
                len(vectors),
                index_name,
            )
