"""Structural chunker — section-aware hybrid strategy.

Uses document headings as primary chunk boundaries, then applies
recursive splitting within large sections. Preserves full heading
hierarchy as parent_context.
"""

from __future__ import annotations

import tiktoken

from smartchunk.models import ChunkerConfig, DocumentSection
from smartchunk.chunkers.base import BaseChunker
from smartchunk.chunkers.recursive import RecursiveChunker


class StructuralChunker(BaseChunker):
    """Chunks documents by their structural heading boundaries.

    Sections that are small enough become a single chunk.
    Sections that exceed ``chunk_size`` are sub-split using
    the :class:`RecursiveChunker`.
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        super().__init__(config)
        self._encoder = tiktoken.get_encoding("cl100k_base")
        self._fallback = RecursiveChunker(config)

    def _token_count(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def chunk(self, sections: list[DocumentSection]) -> list[dict]:
        results: list[dict] = []

        for section in sections:
            parent_context = self._build_parent_context(section)
            text = section.text.strip()

            if not text:
                continue

            token_count = self._token_count(text)

            if token_count <= self.config.chunk_size:
                # Section fits in one chunk
                results.append(
                    {
                        "text": text,
                        "parent_context": parent_context,
                        "page": section.page,
                    }
                )
            else:
                # Section is too large — sub-split with recursive chunker
                sub_section = DocumentSection(
                    text=text,
                    heading=section.heading,
                    level=section.level,
                    page=section.page,
                    parent_headings=section.parent_headings,
                    metadata=section.metadata,
                )
                sub_chunks = self._fallback.chunk([sub_section])
                # Override parent_context with our richer structural version
                for sc in sub_chunks:
                    sc["parent_context"] = parent_context
                results.extend(sub_chunks)

        return results

    def _build_parent_context(self, section: DocumentSection) -> str:
        parts = list(section.parent_headings)
        if section.heading:
            parts.append(section.heading)
        if not parts:
            return ""
        return " → ".join(parts)
