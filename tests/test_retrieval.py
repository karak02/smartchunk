"""Tests for BM25, hybrid retrieval, graph expansion, and reranker pipeline."""

from smartchunk.graph import ChunkGraphBuilder
from smartchunk.models import ChunkMetadata, RetrievalConfig, SmartChunk
from smartchunk.retrieval import BM25Index, HybridRetriever, RerankerPipeline


def test_bm25_retrieval():
    chunks = [
        SmartChunk(
            text="The board approved a $50M capital expansion in Q3.",
            keywords=["capital", "expansion"],
            entities=["$50M"],
            metadata=ChunkMetadata(source="report.txt", chunk_index=0, total_chunks=3, char_count=50, token_count=10),
        ),
        SmartChunk(
            text="Marketing team launched a new brand campaign in summer.",
            keywords=["marketing", "brand"],
            entities=["Marketing"],
            metadata=ChunkMetadata(source="report.txt", chunk_index=1, total_chunks=3, char_count=50, token_count=10),
        ),
        SmartChunk(
            text="Finance and HR operations are standard.",
            keywords=["finance", "operations"],
            entities=[],
            metadata=ChunkMetadata(source="report.txt", chunk_index=2, total_chunks=3, char_count=40, token_count=8),
        ),
    ]

    index = BM25Index(chunks)
    results = index.search("$50M expansion", top_k=2)

    assert len(results) > 0
    assert results[0].chunk.id == chunks[0].id
    assert results[0].bm25_score > 0.0


def test_hybrid_retriever_and_expansion():
    chunks = [
        SmartChunk(
            text="Annual Report 2026 Overview.",
            parent_context="Financials → Executive Summary",
            metadata=ChunkMetadata(source="report.txt", chunk_index=0, total_chunks=3, char_count=28, token_count=5),
        ),
        SmartChunk(
            text="The board approved $50M expansion.",
            parent_context="Financials → Executive Summary",
            metadata=ChunkMetadata(source="report.txt", chunk_index=1, total_chunks=3, char_count=34, token_count=6),
        ),
        SmartChunk(
            text="Risk mitigation plans were reviewed.",
            parent_context="Financials → Risk Analysis",
            metadata=ChunkMetadata(source="report.txt", chunk_index=2, total_chunks=3, char_count=36, token_count=6),
        ),
    ]

    # Build graph
    builder = ChunkGraphBuilder()
    builder.build(chunks)

    config = RetrievalConfig(vector_weight=0.5, bm25_weight=0.5, expand_parents=True, expand_neighbors=1)
    retriever = HybridRetriever(chunks, config=config)

    # Search for $50M
    results = retriever.search("$50M expansion", top_k=1)

    assert len(results) >= 1
    # Check parent/neighbor expansion pulled in context
    retrieved_ids = [r.chunk.id for r in results]
    assert chunks[1].id in retrieved_ids
    # Check siblings/neighbors were expanded
    assert len(results) > 1


def test_custom_reranker():
    chunks = [
        SmartChunk(
            text="Relevant text about AI expansion.",
            metadata=ChunkMetadata(source="doc.txt", chunk_index=0, total_chunks=2, char_count=33, token_count=6),
        ),
        SmartChunk(
            text="Irrelevant text about gardening.",
            metadata=ChunkMetadata(source="doc.txt", chunk_index=1, total_chunks=2, char_count=32, token_count=6),
        ),
    ]

    def mock_reranker(query: str, docs: list[str]) -> list[float]:
        return [0.95 if "AI" in doc else 0.1 for doc in docs]

    pipeline = RerankerPipeline(custom_reranker=mock_reranker)
    retriever = HybridRetriever(chunks, reranker=pipeline)

    results = retriever.search("AI expansion", top_k=2)

    assert len(results) == 2
    assert results[0].chunk.id == chunks[0].id
    assert results[0].rerank_score == 0.95
