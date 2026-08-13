"""Contextual embedding builder — generates optimized text representation for vector indexing.

Combines document provenance, section hierarchy, LLM summary, and raw chunk text
into a rich context string for maximum retrieval accuracy.
"""

from __future__ import annotations

import logging

from smartchunk.models import SmartChunk

logger = logging.getLogger("smartchunk.embedding")


class ContextualEmbeddingBuilder:
    """Builds contextualized text for vector embedding generation."""

    def build(self, chunks: list[SmartChunk], document_title: str = "") -> None:
        """Populate contextual_text on each chunk.

        Parameters
        ----------
        chunks:
            List of SmartChunk objects.
        document_title:
            Optional title or source name of the document.
        """
        for chunk in chunks:
            parts: list[str] = []

            # 1. Document title / source if available
            doc_label = document_title or chunk.metadata.source
            if doc_label and doc_label != "<string>":
                parts.append(f"Document: {doc_label}")

            # 2. Section hierarchy
            if chunk.parent_context:
                parts.append(f"Section: {chunk.parent_context}")

            # 3. LLM Summary
            if chunk.summary:
                parts.append(f"Summary: {chunk.summary}")

            # 4. Raw chunk text
            parts.append(chunk.text)

            chunk.contextual_text = "\n".join(parts)
