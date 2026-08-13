"""SmartChunk — Self-describing document chunks for production RAG.

Every chunk carries enriched metadata — summary, entities, keywords,
parent context — so your retrieval engine doesn't have to guess.

Quick Start::

    from smartchunk import SmartChunker

    chunker = SmartChunker()
    chunks = chunker.process("annual_report.pdf")

    for chunk in chunks:
        print(chunk.summary, chunk.entities, chunk.keywords)
"""

from smartchunk.embedding import ContextualEmbeddingBuilder
from smartchunk.graph import ChunkGraphBuilder
from smartchunk.models import (
    ChunkerConfig,
    ChunkMetadata,
    ChunkRelationship,
    ChunkStrategy,
    DocumentSection,
    EnrichmentConfig,
    EnrichmentField,
    PipelineConfig,
    RelationshipType,
    RetrievalConfig,
    ScoredChunk,
    SmartChunk,
)
from smartchunk.pipeline import SmartChunker
from smartchunk.retrieval import BM25Index, HybridRetriever, RerankerPipeline

__all__ = [
    # Main API & Retrieval
    "SmartChunker",
    "SmartChunk",
    "ScoredChunk",
    "HybridRetriever",
    "BM25Index",
    "RerankerPipeline",
    "ChunkGraphBuilder",
    "ContextualEmbeddingBuilder",
    # Configuration
    "PipelineConfig",
    "ChunkerConfig",
    "EnrichmentConfig",
    "RetrievalConfig",
    "ChunkStrategy",
    "EnrichmentField",
    # Data models & Graph
    "ChunkMetadata",
    "DocumentSection",
    "ChunkRelationship",
    "RelationshipType",
]

__version__ = "0.2.0"
