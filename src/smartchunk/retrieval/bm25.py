"""BM25 lexical index for exact keyword matching and terminology retrieval.

Indexes chunk text, entities, keywords, summary, and parent context.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from smartchunk.models import ScoredChunk, SmartChunk


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """Lexical retrieval index for keyword and exact-match search."""

    def __init__(self, chunks: list[SmartChunk]) -> None:
        self.chunks = chunks
        self.chunk_map = {c.id: c for c in chunks}
        self.corpus_size = len(chunks)
        self.doc_tokens: list[list[str]] = []
        self.doc_len: list[int] = []

        for chunk in chunks:
            # Build search document from text + enriched fields
            search_doc = f"{chunk.text} {' '.join(chunk.keywords)} {' '.join(chunk.entities)} {chunk.summary} {chunk.parent_context}"
            tokens = tokenize(search_doc)
            self.doc_tokens.append(tokens)
            self.doc_len.append(len(tokens))

        self.avg_doc_len = sum(self.doc_len) / max(1, self.corpus_size)

        # Try to use rank_bm25 if available, else fallback to internal implementation
        self._bm25_model: Any = None
        try:
            from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

            self._bm25_model = BM25Okapi(self.doc_tokens)
        except ImportError:
            self._build_internal_index()

    def _build_internal_index(self) -> None:
        """Internal fallback BM25 stats."""
        self.df: Counter[str] = Counter()
        self.doc_freqs: list[Counter[str]] = []

        for tokens in self.doc_tokens:
            counts = Counter(tokens)
            self.doc_freqs.append(counts)
            for word in counts:
                self.df[word] += 1

    def search(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        """Search the BM25 index for query terms.

        Parameters
        ----------
        query:
            Search string.
        top_k:
            Number of top candidates to return.

        Returns
        -------
        list[ScoredChunk]
            Ordered candidates with bm25_score populated.
        """
        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        if self._bm25_model is not None:
            scores = self._bm25_model.get_scores(query_tokens)
        else:
            scores = self._compute_internal_scores(query_tokens)

        # Normalize scores to [0, 1]
        max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0

        results: list[ScoredChunk] = []
        for idx, score in enumerate(scores):
            if score > 0:
                norm_score = float(score / max_score)
                chunk = self.chunks[idx]
                results.append(
                    ScoredChunk(
                        chunk=chunk,
                        score=norm_score,
                        bm25_score=norm_score,
                    )
                )

        results.sort(key=lambda s: s.bm25_score, reverse=True)
        return results[:top_k]

    def _compute_internal_scores(self, query_tokens: list[str]) -> list[float]:
        """Fallback BM25 score computation."""
        k1 = 1.5
        b = 0.75
        scores = [0.0] * self.corpus_size

        for token in query_tokens:
            if token not in self.df:
                continue
            # IDF calculation
            n = self.df[token]
            idf = math.log((self.corpus_size - n + 0.5) / (n + 0.5) + 1.0)

            for i, counts in enumerate(self.doc_freqs):
                tf = counts[token]
                if tf > 0:
                    denom = tf + k1 * (1.0 - b + b * (self.doc_len[i] / self.avg_doc_len))
                    scores[i] += idf * (tf * (k1 + 1.0)) / denom

        return scores
