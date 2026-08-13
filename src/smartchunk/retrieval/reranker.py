"""Reranking integration point for SmartChunk.

Wraps CrossEncoder or custom rerankers to re-score retrieval candidates.
"""

from __future__ import annotations

import logging
from typing import Callable

from smartchunk.models import ScoredChunk

logger = logging.getLogger("smartchunk.retrieval.reranker")


class RerankerPipeline:
    """Reranker wrapper supporting CrossEncoders and custom scoring callables."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        custom_reranker: Callable[[str, list[str]], list[float]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.custom_reranker = custom_reranker
        self._cross_encoder = None

    def _get_cross_encoder(self):
        """Lazy load CrossEncoder from sentence-transformers."""
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

                self._cross_encoder = CrossEncoder(self.model_name)
            except ImportError as exc:
                raise ImportError(
                    "Reranking with sentence-transformers requires sentence-transformers. "
                    "Install with: pip install smartchunk[reranker]"
                ) from exc
        return self._cross_encoder

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int = 5,
    ) -> list[ScoredChunk]:
        """Rerank candidates using cross-encoder or custom reranker.

        Parameters
        ----------
        query:
            User search query string.
        candidates:
            Candidates from initial retrieval.
        top_k:
            Number of top candidates to return.

        Returns
        -------
        list[ScoredChunk]
            Reranked and truncated list of scored chunks.
        """
        if not candidates or not query:
            return candidates[:top_k]

        texts = [c.chunk.contextual_text or c.chunk.text for c in candidates]

        if self.custom_reranker is not None:
            scores = self.custom_reranker(query, texts)
        else:
            cross_encoder = self._get_cross_encoder()
            pairs = [[query, text] for text in texts]
            raw_scores = cross_encoder.predict(pairs)
            scores = [float(s) for s in raw_scores]

        # Update candidates with rerank_score
        for candidate, r_score in zip(candidates, scores):
            candidate.rerank_score = r_score
            candidate.score = r_score

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]
