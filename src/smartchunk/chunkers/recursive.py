"""Recursive text splitter — the reliable fallback chunking strategy.

Splits text using a hierarchy of separators (paragraphs → lines → sentences → words)
and respects chunk_size / overlap settings.
"""

from __future__ import annotations

import tiktoken

from smartchunk.chunkers.base import BaseChunker
from smartchunk.models import ChunkerConfig, DocumentSection

# Separator hierarchy: try the most structural first, fall back to finer splits.
_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]


class RecursiveChunker(BaseChunker):
    """Recursively splits text using a hierarchy of separators.

    This is the default / fallback strategy that works for any text.
    It prioritises splitting on paragraph breaks, then sentence boundaries,
    and respects the configured ``chunk_size`` (in tokens) and ``chunk_overlap``.
    """

    def __init__(
        self,
        config: ChunkerConfig | None = None,
        separators: list[str] | None = None,
    ) -> None:
        super().__init__(config)
        self.separators = separators or _DEFAULT_SEPARATORS
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def _token_count(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self._encoder.encode(text))

    def chunk(self, sections: list[DocumentSection]) -> list[dict]:
        results: list[dict] = []

        for section in sections:
            parent_context = self._build_parent_context(section)

            if section.content_type == "table" and section.table:
                # Keep table intact if possible
                table_md = section.table.to_markdown() or section.text
                results.append(
                    {
                        "text": table_md.strip(),
                        "parent_context": parent_context,
                        "page": section.page,
                        "content_type": "table",
                        "table": section.table,
                        "figures": section.figures,
                    }
                )
                continue

            text_chunks = self._recursive_split(section.text, 0)

            for chunk_text in text_chunks:
                if chunk_text.strip():
                    results.append(
                        {
                            "text": chunk_text.strip(),
                            "parent_context": parent_context,
                            "page": section.page,
                            "content_type": section.content_type,
                            "table": section.table,
                            "figures": section.figures,
                        }
                    )

        return results

    def _build_parent_context(self, section: DocumentSection) -> str:
        """Build a parent context string from section heading hierarchy."""
        parts = list(section.parent_headings)
        if section.heading:
            parts.append(section.heading)
        if not parts:
            return ""
        return " → ".join(parts)

    def _recursive_split(self, text: str, sep_index: int) -> list[str]:
        """Recursively split text using the separator hierarchy."""
        if self._token_count(text) <= self.config.chunk_size:
            return [text] if text.strip() else []

        if sep_index >= len(self.separators):
            # Last resort: hard-split by tokens
            return self._hard_split(text)

        separator = self.separators[sep_index]
        parts = text.split(separator)

        if len(parts) <= 1:
            # This separator didn't help, try the next one
            return self._recursive_split(text, sep_index + 1)

        chunks: list[str] = []
        current_chunk = ""

        for part in parts:
            candidate = current_chunk + separator + part if current_chunk else part

            if self._token_count(candidate) <= self.config.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    # Current chunk is full — try to recursively split if it's still too big
                    if self._token_count(current_chunk) > self.config.chunk_size:
                        chunks.extend(self._recursive_split(current_chunk, sep_index + 1))
                    else:
                        chunks.append(current_chunk)

                # Start new chunk; if this single part is too big, recurse deeper
                if self._token_count(part) > self.config.chunk_size:
                    chunks.extend(self._recursive_split(part, sep_index + 1))
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk:
            if self._token_count(current_chunk) > self.config.chunk_size:
                chunks.extend(self._recursive_split(current_chunk, sep_index + 1))
            else:
                chunks.append(current_chunk)

        # Apply overlap
        if self.config.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _hard_split(self, text: str) -> list[str]:
        """Last-resort token-level splitting."""
        tokens = self._encoder.encode(text)
        chunks: list[str] = []
        step = max(1, self.config.chunk_size - self.config.chunk_overlap)

        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + self.config.chunk_size]
            chunk_text = self._encoder.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text)

        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap tokens from the end of the previous chunk to the start of the next."""
        if self.config.chunk_overlap <= 0:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tokens = self._encoder.encode(chunks[i - 1])
            overlap_tokens = prev_tokens[-self.config.chunk_overlap :]
            overlap_text = self._encoder.decode(overlap_tokens)
            result.append(overlap_text + " " + chunks[i])

        return result
