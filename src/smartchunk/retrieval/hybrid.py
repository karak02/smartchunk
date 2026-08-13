"""Hybrid retriever — combines dense vector retrieval, BM25 lexical search, metadata filtering, parent-child expansion, and neighbor graph walking.
"""

from __future__ import annotations

import logging
from typing import Callable

from smartchunk.models import RetrievalConfig, ScoredChunk, SmartChunk
from smartchunk.retrieval.bm25 import BM25Index
from smartchunk.retrieval.reranker import RerankerPipeline

logger = logging.getLogger("smartchunk.retrieval.hybrid")


class HybridRetriever:
    """Hybrid retrieval engine combining BM25, dense search, and graph expansion."""

    def __init__(
        self,
        chunks: list[SmartChunk],
        config: RetrievalConfig | None = None,
        dense_search_fn: Callable[[str, int], list[ScoredChunk]] | None = None,
        reranker: RerankerPipeline | None = None,
    ) -> None:
        self.chunks = chunks
        self.chunk_map = {c.id: c for c in chunks}
        self.config = config or RetrievalConfig()
        self.bm25_index = BM25Index(chunks)
        self.dense_search_fn = dense_search_fn
        self.reranker = reranker

        if self.reranker is None and self.config.reranker_model:
            self.reranker = RerankerPipeline(model_name=self.config.reranker_model)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, str | int | float] | None = None,
        expand_parents: bool | None = None,
        expand_neighbors: int | None = None,
    ) -> list[ScoredChunk]:
        """Perform hybrid search over the document chunk collection.

        Parameters
        ----------
        query:
            Search string.
        top_k:
            Number of final candidates to return.
        filters:
            Metadata filtering key-value pairs (e.g. {"source": "report.pdf"}).
        expand_parents:
            Override config expand_parents setting.
        expand_neighbors:
            Override config expand_neighbors setting.

        Returns
        -------
        list[ScoredChunk]
            Ordered, scored, and expanded candidate chunks.
        """
        if not self.chunks or not query.strip():
            return []

        should_expand_parents = expand_parents if expand_parents is not None else self.config.expand_parents
        num_expand_neighbors = expand_neighbors if expand_neighbors is not None else self.config.expand_neighbors

        # 1. Lexical BM25 Candidates
        bm25_candidates = self.bm25_index.search(query, top_k=top_k * 3)

        # 2. Dense Vector Candidates
        dense_candidates: list[ScoredChunk] = []
        if self.dense_search_fn is not None:
            dense_candidates = self.dense_search_fn(query, top_k * 3)
        else:
            dense_candidates = self._default_dense_search(query, top_k * 3)

        # 3. Reciprocal Rank Fusion (RRF)
        combined = self._reciprocal_rank_fusion(
            bm25_candidates=bm25_candidates,
            dense_candidates=dense_candidates,
            vector_weight=self.config.vector_weight,
            bm25_weight=self.config.bm25_weight,
        )

        # 4. Metadata Filtering
        if filters:
            combined = [sc for sc in combined if self._matches_filter(sc.chunk, filters)]

        # Truncate to top_k before expansion
        candidates = combined[:top_k]

        # 5. Reranking stage (if configured)
        if self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates, top_k=self.config.reranker_top_k)

        # 6. Graph Expansions (Parent-Child & Neighbor Walking)
        if should_expand_parents or num_expand_neighbors > 0:
            candidates = self._expand_graph_context(
                candidates,
                expand_parents=should_expand_parents,
                expand_neighbors=num_expand_neighbors,
            )

        return candidates

    def _default_dense_search(self, query: str, top_k: int) -> list[ScoredChunk]:
        """Fallback dense similarity search using sentence-transformers if installed."""
        try:
            import numpy as np  # type: ignore[import-untyped]
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            model = SentenceTransformer("all-MiniLM-L6-v2")
            doc_texts = [c.contextual_text or c.text for c in self.chunks]
            doc_embeddings = model.encode(doc_texts, show_progress_bar=False)
            query_embedding = model.encode(query, show_progress_bar=False)

            scores: list[float] = []
            for emb in doc_embeddings:
                sim = float(np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-10))
                scores.append(max(0.0, sim))

            results: list[ScoredChunk] = []
            for idx, score in enumerate(scores):
                if score > 0:
                    results.append(
                        ScoredChunk(
                            chunk=self.chunks[idx],
                            score=score,
                            dense_score=score,
                        )
                    )

            results.sort(key=lambda s: s.dense_score, reverse=True)
            return results[:top_k]

        except ImportError:
            # Sentence-transformers not available — fall back gracefully to BM25 scores
            return []

    def _reciprocal_rank_fusion(
        self,
        bm25_candidates: list[ScoredChunk],
        dense_candidates: list[ScoredChunk],
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        k: int = 60,
    ) -> list[ScoredChunk]:
        """Combine ranks using Reciprocal Rank Fusion (RRF)."""
        rrf_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}
        dense_scores: dict[str, float] = {}

        # Dense ranks
        for rank, sc in enumerate(dense_candidates):
            cid = sc.chunk.id
            dense_scores[cid] = sc.dense_score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + vector_weight / (k + rank + 1)

        # BM25 ranks
        for rank, sc in enumerate(bm25_candidates):
            cid = sc.chunk.id
            bm25_scores[cid] = sc.bm25_score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + bm25_weight / (k + rank + 1)

        # Build final ScoredChunks list
        fused: list[ScoredChunk] = []
        for cid, score in rrf_scores.items():
            chunk = self.chunk_map[cid]
            fused.append(
                ScoredChunk(
                    chunk=chunk,
                    score=score,
                    dense_score=dense_scores.get(cid, 0.0),
                    bm25_score=bm25_scores.get(cid, 0.0),
                )
            )

        fused.sort(key=lambda s: s.score, reverse=True)
        return fused

    def _matches_filter(self, chunk: SmartChunk, filters: dict[str, str | int | float]) -> bool:
        """Check if chunk metadata matches filter criteria."""
        for key, expected in filters.items():
            # Check direct chunk attributes
            if hasattr(chunk, key):
                val = getattr(chunk, key)
                if str(val).lower() != str(expected).lower():
                    return False
            # Check metadata attributes
            elif hasattr(chunk.metadata, key):
                val = getattr(chunk.metadata, key)
                if str(val).lower() != str(expected).lower():
                    return False
            else:
                return False
        return True

    def _expand_graph_context(
        self,
        candidates: list[ScoredChunk],
        expand_parents: bool,
        expand_neighbors: int,
    ) -> list[ScoredChunk]:
        """Expand retrieved candidates with parent section siblings and neighbor nodes."""
        seen_ids = {sc.chunk.id for sc in candidates}
        expanded_list = list(candidates)

        for sc in candidates:
            chunk = sc.chunk

            # Parent section expansion: find all chunks sharing section_id
            if expand_parents and chunk.section_id:
                for other in self.chunks:
                    if other.section_id == chunk.section_id and other.id not in seen_ids:
                        seen_ids.add(other.id)
                        expanded_list.append(
                            ScoredChunk(
                                chunk=other,
                                score=sc.score * 0.9,  # slightly lower score for expanded context
                                dense_score=sc.dense_score,
                                bm25_score=sc.bm25_score,
                            )
                        )

            # Neighbor walking (prev_id / next_id)
            if expand_neighbors > 0:
                # Walk backwards
                curr = chunk
                for _ in range(expand_neighbors):
                    if curr.prev_id and curr.prev_id in self.chunk_map:
                        prev_chunk = self.chunk_map[curr.prev_id]
                        if prev_chunk.id not in seen_ids:
                            seen_ids.add(prev_chunk.id)
                            expanded_list.append(
                                ScoredChunk(
                                    chunk=prev_chunk,
                                    score=sc.score * 0.85,
                                )
                            )
                        curr = prev_chunk
                    else:
                        break

                # Walk forwards
                curr = chunk
                for _ in range(expand_neighbors):
                    if curr.next_id and curr.next_id in self.chunk_map:
                        next_chunk = self.chunk_map[curr.next_id]
                        if next_chunk.id not in seen_ids:
                            seen_ids.add(next_chunk.id)
                            expanded_list.append(
                                ScoredChunk(
                                    chunk=next_chunk,
                                    score=sc.score * 0.85,
                                )
                            )
                        curr = next_chunk
                    else:
                        break

        # Re-sort to preserve original document order or ranking
        return expanded_list
