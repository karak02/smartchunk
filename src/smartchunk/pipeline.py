"""SmartChunker — the main pipeline orchestrator.

Ties together parsing, chunking, enrichment, and export into
a single coherent API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import tiktoken

from smartchunk.chunkers import get_chunker
from smartchunk.config import load_config
from smartchunk.embedding import ContextualEmbeddingBuilder
from smartchunk.enrichment.llm import LLMEnricher
from smartchunk.exporters.chromadb import ChromaExporter
from smartchunk.exporters.json_export import JsonExporter
from smartchunk.exporters.pinecone import PineconeExporter
from smartchunk.graph import ChunkGraphBuilder
from smartchunk.models import (
    ChunkMetadata,
    ChunkStrategy,
    EnrichmentField,
    PipelineConfig,
    SmartChunk,
)
from smartchunk.parsers import get_parser

logger = logging.getLogger("smartchunk")


class SmartChunker:
    """Self-describing document chunker for production RAG.

    Process documents into enriched chunks with summaries, entities,
    keywords, parent context, and neighbor summaries.

    Examples
    --------
    Basic usage::

        from smartchunk import SmartChunker

        chunker = SmartChunker()
        chunks = chunker.process("report.pdf")

        for chunk in chunks:
            print(chunk.summary)
            print(chunk.entities)

    Configured usage::

        chunker = SmartChunker(
            model="gpt-4o-mini",
            chunk_size=512,
            strategy="semantic",
            enrich=True,
        )

    Export::

        chunker.to_json(chunks, "output.json")
        chunker.to_pinecone(chunks, index_name="my-index")
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        strategy: str | ChunkStrategy | None = None,
        enrich: bool | None = None,
        enrichments: list[str | EnrichmentField] | None = None,
        batch_size: int | None = None,
        max_concurrency: int | None = None,
        temperature: float | None = None,
        similarity_threshold: float | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        """Initialise the SmartChunker pipeline.

        Parameters
        ----------
        model:
            LLM model string (any LiteLLM-supported model).
        chunk_size:
            Target chunk size in tokens.
        chunk_overlap:
            Overlap between consecutive chunks in tokens.
        strategy:
            Chunking strategy: "semantic", "recursive", or "structural".
        enrich:
            Whether to run LLM enrichment.
        enrichments:
            Which enrichment fields to populate.
        batch_size:
            Number of chunks to enrich per batch.
        max_concurrency:
            Max concurrent LLM calls.
        temperature:
            LLM sampling temperature.
        similarity_threshold:
            Cosine similarity threshold for semantic chunking.
        config:
            Pre-built PipelineConfig (overrides all other params).
        """
        if config:
            self._config = config
        else:
            overrides: dict[str, Any] = {}
            if model is not None:
                overrides["model"] = model
            if chunk_size is not None:
                overrides["chunk_size"] = chunk_size
            if chunk_overlap is not None:
                overrides["chunk_overlap"] = chunk_overlap
            if strategy is not None:
                overrides["strategy"] = strategy
            if enrich is not None:
                overrides["enrich"] = enrich
            if enrichments is not None:
                overrides["enrichments"] = enrichments
            if batch_size is not None:
                overrides["batch_size"] = batch_size
            if max_concurrency is not None:
                overrides["max_concurrency"] = max_concurrency
            if temperature is not None:
                overrides["temperature"] = temperature
            if similarity_threshold is not None:
                overrides["similarity_threshold"] = similarity_threshold

            self._config = load_config(**overrides)

        self._encoder = tiktoken.get_encoding("cl100k_base")
        self._enricher: LLMEnricher | None = None

    @property
    def config(self) -> PipelineConfig:
        """The resolved pipeline configuration."""
        return self._config

    @property
    def usage_stats(self) -> dict[str, Any]:
        """Cumulative LLM usage stats (tokens, estimated cost)."""
        if self._enricher:
            return self._enricher.usage_stats
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0}

    @property
    def cache_stats(self) -> dict[str, Any]:
        """Cache performance metrics (hits, misses, savings)."""
        if self._enricher:
            return self._enricher.cache_stats
        return {
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_entries": 0,
            "hit_rate_percent": 0.0,
            "llm_calls_saved": 0,
            "estimated_time_saved_seconds": 0.0,
        }

    # ── Main Processing ────────────────────────────────────────────────────────

    def process(self, filepath: str | Path) -> list[SmartChunk]:
        """Process a document file into enriched SmartChunks.

        Parameters
        ----------
        filepath:
            Path to the document (PDF, TXT, MD).

        Returns
        -------
        list[SmartChunk]
            Ordered list of self-describing chunks.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        logger.info("Processing file: %s", filepath.name)

        # Step 1: Parse
        parser = get_parser(filepath)
        sections = parser.parse(filepath)
        logger.info("Parsed %d sections from %s", len(sections), filepath.name)

        # Step 2: Chunk
        chunker = get_chunker(self._config.chunker)
        raw_chunks = chunker.chunk(sections)
        logger.info("Created %d raw chunks", len(raw_chunks))

        # Step 3: Build SmartChunk objects
        source_name = filepath.name
        chunks = self._build_smart_chunks(raw_chunks, source_name)

        # Step 3.5: Build document graph (sequential, section, depth)
        graph_builder = ChunkGraphBuilder()
        graph_builder.build(chunks)

        # Step 4: Enrich (if enabled)
        if self._config.enrich:
            self._enricher = LLMEnricher(self._config.enrichment)
            self._enricher.enrich(chunks)
            logger.info("Enrichment complete. %s", self._enricher.usage_stats)

            # Link shared entity relationships post-enrichment
            graph_builder.link_entities(chunks)

        # Step 5: Build contextual embeddings text
        embedding_builder = ContextualEmbeddingBuilder()
        embedding_builder.build(chunks, document_title=filepath.stem)

        return chunks

    def process_text(
        self,
        text: str,
        source: str = "<string>",
    ) -> list[SmartChunk]:
        """Process raw text into enriched SmartChunks.

        Parameters
        ----------
        text:
            Raw text to process.
        source:
            Label for provenance metadata.

        Returns
        -------
        list[SmartChunk]
            Ordered list of self-describing chunks.
        """
        logger.info("Processing text from source: %s", source)

        # Use text parser for raw strings
        from smartchunk.parsers.text import TextParser

        parser = TextParser()
        sections = parser.parse_text(text, source=source)

        chunker = get_chunker(self._config.chunker)
        raw_chunks = chunker.chunk(sections)

        chunks = self._build_smart_chunks(raw_chunks, source)

        # Build document graph
        graph_builder = ChunkGraphBuilder()
        graph_builder.build(chunks)

        if self._config.enrich:
            self._enricher = LLMEnricher(self._config.enrichment)
            self._enricher.enrich(chunks)
            logger.info("Enrichment complete. %s", self._enricher.usage_stats)

            # Link shared entity relationships post-enrichment
            graph_builder.link_entities(chunks)

        # Build contextual embeddings text
        embedding_builder = ContextualEmbeddingBuilder()
        embedding_builder.build(chunks, document_title=source)

        return chunks

    # ── Export Shortcuts ───────────────────────────────────────────────────────

    @staticmethod
    def to_json(chunks: list[SmartChunk], filepath: str | Path) -> None:
        """Export chunks to a pretty-printed JSON file."""
        JsonExporter.to_json(chunks, filepath)
        logger.info("Exported %d chunks to %s", len(chunks), filepath)

    @staticmethod
    def to_jsonl(chunks: list[SmartChunk], filepath: str | Path) -> None:
        """Export chunks to a JSONL file."""
        JsonExporter.to_jsonl(chunks, filepath)
        logger.info("Exported %d chunks to %s (JSONL)", len(chunks), filepath)

    @staticmethod
    def to_dict(chunks: list[SmartChunk]) -> list[dict[str, Any]]:
        """Convert chunks to a list of plain dictionaries."""
        return JsonExporter.to_dict(chunks)

    @staticmethod
    def to_pinecone(
        chunks: list[SmartChunk],
        index_name: str,
        api_key: str | None = None,
        namespace: str = "",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        """Export chunks to a Pinecone index."""
        exporter = PineconeExporter(
            api_key=api_key,
            index_name=index_name,
            embedding_model=embedding_model,
        )
        exporter.to_pinecone(chunks, index_name=index_name, api_key=api_key, namespace=namespace)

    @staticmethod
    def to_chromadb(
        chunks: list[SmartChunk],
        collection: str = "smartchunk",
        persist_directory: str | Path | None = None,
    ) -> None:
        """Export chunks to a ChromaDB collection."""
        exporter = ChromaExporter(
            collection_name=collection,
            persist_directory=persist_directory,
        )
        exporter.to_chromadb(chunks, collection_name=collection, persist_directory=persist_directory)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _build_smart_chunks(
        self,
        raw_chunks: list[dict],
        source: str,
    ) -> list[SmartChunk]:
        """Convert raw chunk dicts into SmartChunk objects with metadata."""
        total = len(raw_chunks)
        smart_chunks: list[SmartChunk] = []

        for i, raw in enumerate(raw_chunks):
            text = raw["text"]
            token_count = len(self._encoder.encode(text))

            chunk = SmartChunk(
                text=text,
                parent_context=raw.get("parent_context", ""),
                content_type=raw.get("content_type", "text"),
                table=raw.get("table"),
                figures=raw.get("figures", []),
                strategy=self._config.chunker.strategy.value,
                cache_status="DISABLED" if not self._config.enrich else "MISS",
                metadata=ChunkMetadata(
                    source=source,
                    page=raw.get("page"),
                    chunk_index=i,
                    total_chunks=total,
                    char_count=len(text),
                    token_count=token_count,
                ),
            )
            smart_chunks.append(chunk)

        return smart_chunks
