"""Semantic chunker — embedding-based topic boundary detection.

Requires the ``semantic`` extra::

    pip install smartchunk[semantic]
"""

from __future__ import annotations

import tiktoken

from smartchunk.models import ChunkerConfig, DocumentSection
from smartchunk.chunkers.base import BaseChunker


class SemanticChunker(BaseChunker):
    """Splits text by detecting semantic topic boundaries.

    Uses sentence-transformers to embed individual sentences, then
    finds chunk boundaries where cosine similarity between adjacent
    sentences drops below a configurable threshold.

    Falls back to the recursive chunker for sections that are too small
    to benefit from semantic analysis.
    """

    _model = None  # Lazy-loaded sentence transformer

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        super().__init__(config)
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def _get_model(self):
        """Lazy-load sentence transformer model."""
        if SemanticChunker._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "Semantic chunking requires sentence-transformers. "
                    "Install with: pip install smartchunk[semantic]"
                ) from exc
            SemanticChunker._model = SentenceTransformer("all-MiniLM-L6-v2")
        return SemanticChunker._model

    def _token_count(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def _split_into_sentences(self, text: str) -> list[str]:
        """Simple sentence splitting using regex."""
        import re

        # Split on sentence-ending punctuation followed by space/newline
        raw = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw if s.strip()]

    def chunk(self, sections: list[DocumentSection]) -> list[dict]:
        results: list[dict] = []

        for section in sections:
            parent_context = self._build_parent_context(section)
            sentences = self._split_into_sentences(section.text)

            if len(sentences) <= 2:
                # Too few sentences for semantic analysis
                if section.text.strip():
                    results.append(
                        {
                            "text": section.text.strip(),
                            "parent_context": parent_context,
                            "page": section.page,
                        }
                    )
                continue

            # Compute semantic boundaries
            boundaries = self._find_boundaries(sentences)
            chunks = self._merge_by_boundaries(sentences, boundaries)

            for chunk_text in chunks:
                if chunk_text.strip():
                    results.append(
                        {
                            "text": chunk_text.strip(),
                            "parent_context": parent_context,
                            "page": section.page,
                        }
                    )

        return results

    def _find_boundaries(self, sentences: list[str]) -> list[int]:
        """Find sentence indices where the topic shifts.

        Returns a sorted list of indices where a new chunk should start.
        """
        import numpy as np  # type: ignore[import-untyped]

        model = self._get_model()
        embeddings = model.encode(sentences, show_progress_bar=False)

        # Compute cosine similarity between consecutive sentences
        similarities: list[float] = []
        for i in range(len(embeddings) - 1):
            a = embeddings[i]
            b = embeddings[i + 1]
            cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
            similarities.append(cos_sim)

        # Find boundaries where similarity drops below threshold
        boundaries = [0]  # Always start a chunk at index 0
        for i, sim in enumerate(similarities):
            if sim < self.config.similarity_threshold:
                boundaries.append(i + 1)

        return boundaries

    def _merge_by_boundaries(self, sentences: list[str], boundaries: list[int]) -> list[str]:
        """Merge sentences between boundaries into chunks, respecting max chunk size."""
        chunks: list[str] = []
        boundary_set = sorted(set(boundaries))

        for i, start in enumerate(boundary_set):
            end = boundary_set[i + 1] if i + 1 < len(boundary_set) else len(sentences)
            chunk_text = " ".join(sentences[start:end])

            # If this merged chunk is too large, sub-split it
            if self._token_count(chunk_text) > self.config.chunk_size:
                sub_chunks = self._size_split(sentences[start:end])
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text)

        return chunks

    def _size_split(self, sentences: list[str]) -> list[str]:
        """Split a group of sentences into chunks that respect max token size."""
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self._token_count(sentence)

            if current_tokens + sent_tokens > self.config.chunk_size and current:
                chunks.append(" ".join(current))
                # Apply overlap: keep last few tokens worth of sentences
                overlap_tokens = 0
                overlap_sents: list[str] = []
                for s in reversed(current):
                    t = self._token_count(s)
                    if overlap_tokens + t > self.config.chunk_overlap:
                        break
                    overlap_sents.insert(0, s)
                    overlap_tokens += t
                current = overlap_sents
                current_tokens = overlap_tokens

            current.append(sentence)
            current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _build_parent_context(self, section: DocumentSection) -> str:
        parts = list(section.parent_headings)
        if section.heading:
            parts.append(section.heading)
        return " → ".join(parts) if parts else ""
