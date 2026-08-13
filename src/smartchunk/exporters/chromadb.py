"""ChromaDB vector database exporter.

Requires the ``chromadb`` extra::

    pip install smartchunk[chromadb]
"""

from __future__ import annotations

import logging
from pathlib import Path

from smartchunk.exporters.base import BaseExporter
from smartchunk.models import SmartChunk

logger = logging.getLogger("smartchunk.exporters.chromadb")


class ChromaExporter(BaseExporter):
    """Exports SmartChunks to a ChromaDB collection.

    ChromaDB handles embedding internally (default: all-MiniLM-L6-v2)
    so no external embedding API call is needed.
    """

    def __init__(
        self,
        collection_name: str = "smartchunk",
        persist_directory: str | Path | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = str(persist_directory) if persist_directory else None

    def export(self, chunks: list[SmartChunk], **kwargs: object) -> None:
        """Export chunks to ChromaDB."""
        collection_name = str(kwargs.get("collection", self.collection_name))
        self.to_chromadb(chunks, collection_name=collection_name)

    def to_chromadb(
        self,
        chunks: list[SmartChunk],
        collection_name: str | None = None,
        persist_directory: str | Path | None = None,
    ) -> None:
        """Upload chunks to a ChromaDB collection.

        Parameters
        ----------
        chunks:
            List of SmartChunks to upload.
        collection_name:
            Name for the ChromaDB collection.
        persist_directory:
            Directory for persistent storage. If ``None``, uses in-memory.
        """
        try:
            import chromadb  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "ChromaDB export requires chromadb. Install with: pip install smartchunk[chromadb]"
            ) from exc

        collection_name = collection_name or self.collection_name
        persist_dir = persist_directory or self.persist_directory

        if persist_dir:
            client = chromadb.PersistentClient(path=str(persist_dir))
        else:
            client = chromadb.Client()

        collection = client.get_or_create_collection(name=collection_name)

        # Prepare data for batch add
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for chunk in chunks:
            ids.append(chunk.id)
            documents.append(chunk.contextual_text or chunk.text)
            metadatas.append(
                {
                    "raw_text": chunk.text,
                    "summary": chunk.summary,
                    "entities": ", ".join(
                        chunk.entities
                    ),  # ChromaDB metadata must be str/int/float
                    "keywords": ", ".join(chunk.keywords),
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
            )

        # ChromaDB supports batch operations
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            collection.upsert(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

        logger.info(
            "Upserted %d documents to ChromaDB collection '%s'",
            len(chunks),
            collection_name,
        )
