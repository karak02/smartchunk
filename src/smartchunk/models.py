"""Core data models for SmartChunk.

All public data structures are Pydantic models for validation,
serialization, and JSON schema generation.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ── Enums ──────────────────────────────────────────────────────────────────────


class ChunkStrategy(str, Enum):
    """Available chunking strategies."""

    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    STRUCTURAL = "structural"


class EnrichmentField(str, Enum):
    """Individual enrichment fields that can be toggled."""

    SUMMARY = "summary"
    ENTITIES = "entities"
    KEYWORDS = "keywords"
    CONFIDENCE = "confidence"


class RelationshipType(str, Enum):
    """Types of relationships between chunks in a document graph."""

    FOLLOWS = "follows"
    PRECEDES = "precedes"
    SAME_SECTION = "same_section"
    PARENT = "parent"
    CHILD = "child"
    SHARED_ENTITY = "shared_entity"


class ChunkRelationship(BaseModel):
    """Edge in the chunk graph connecting two chunks."""

    target_id: str = Field(..., description="ID of the related target chunk.")
    relation: RelationshipType = Field(..., description="Type of relationship.")
    weight: float = Field(default=1.0, description="Relationship weight/strength (0.0 to 1.0).")


# ── Document Representation & Structures ────────────────────────────────────────


class TableData(BaseModel):
    """Structured representation of a document table."""

    headers: list[str] = Field(default_factory=list, description="Column header titles.")
    rows: list[list[str]] = Field(
        default_factory=list, description="2D matrix of cell text values."
    )
    caption: str = Field(default="", description="Table title or caption.")

    def to_markdown(self) -> str:
        """Format table as a markdown string."""
        if not self.headers and not self.rows:
            return ""

        lines: list[str] = []
        if self.caption:
            lines.append(f"**Table: {self.caption}**\n")

        headers = (
            self.headers
            if self.headers
            else [f"Col {i + 1}" for i in range(len(self.rows[0]) if self.rows else 0)]
        )
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for row in self.rows:
            # Pad or truncate row to header length
            cells = [row[i] if i < len(row) else "" for i in range(len(headers))]
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def to_text(self) -> str:
        """Linearize table as natural language sentences for search/embedding."""
        if not self.rows:
            return self.caption

        headers = (
            self.headers if self.headers else [f"Field_{i + 1}" for i in range(len(self.rows[0]))]
        )
        items: list[str] = []

        if self.caption:
            items.append(f"Table ({self.caption}):")

        for row in self.rows:
            pairs = [f"{h}: {val}" for h, val in zip(headers, row) if val.strip()]
            if pairs:
                items.append(", ".join(pairs))

        return " | ".join(items)


class FigureRef(BaseModel):
    """Reference to an image, chart, or figure in the document."""

    image_path: str = Field(default="", description="Path or URI to extracted image file.")
    caption: str = Field(default="", description="Figure title or caption.")
    page: int | None = Field(default=None, description="Page number (1-indexed).")
    bbox: tuple[float, float, float, float] | None = Field(
        default=None,
        description="Bounding box coordinates (x0, y0, x1, y1) on page.",
    )


class DocumentSection(BaseModel):
    """Intermediate representation of a parsed document section.

    Produced by parsers, consumed by chunkers.
    """

    text: str = Field(..., description="Raw text content of this section.")
    heading: str | None = Field(default=None, description="Heading/title of this section, if any.")
    level: int = Field(
        default=0,
        ge=0,
        description="Heading depth (0 = root/no heading, 1 = H1, 2 = H2, …).",
    )
    page: int | None = Field(
        default=None, description="Page number (1-indexed), if sourced from PDF."
    )
    parent_headings: list[str] = Field(
        default_factory=list,
        description="Ancestor heading chain, outermost first. E.g. ['Chapter 1', 'Budget'].",
    )
    content_type: str = Field(
        default="text",
        description="Type of content: 'text', 'table', or 'figure'.",
    )
    table: TableData | None = Field(
        default=None,
        description="Structured table data if content_type is 'table'.",
    )
    figures: list[FigureRef] = Field(
        default_factory=list,
        description="Figures or images associated with this section.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extra metadata from the parser.",
    )


# ── Chunk Output ───────────────────────────────────────────────────────────────


class ChunkMetadata(BaseModel):
    """Provenance and sizing metadata attached to every chunk."""

    source: str = Field(..., description="Origin filename or identifier.")
    page: int | None = Field(default=None, description="Page number (1-indexed).")
    chunk_index: int = Field(..., description="0-based position within the document.")
    total_chunks: int = Field(..., description="Total number of chunks in the document.")
    char_count: int = Field(..., description="Number of characters in chunk text.")
    token_count: int = Field(default=0, description="Estimated token count (tiktoken).")


class SmartChunk(BaseModel):
    """A self-describing chunk — the primary output of SmartChunker.

    Every chunk carries enriched metadata so your retrieval engine
    doesn't have to guess.
    """

    id: str = Field(
        default_factory=lambda: f"chunk_{uuid.uuid4().hex[:12]}",
        description="Unique chunk identifier.",
    )
    text: str = Field(..., description="The raw chunk text.")

    # ── Contextual Embedding ──
    contextual_text: str = Field(
        default="",
        description=(
            "Pre-built contextual text for vector embeddings "
            "(document context + summary + raw text)."
        ),
    )
    cache_status: str = Field(
        default="MISS", description="Enrichment cache status ('HIT', 'MISS', or 'DISABLED')."
    )
    strategy: str = Field(
        default="recursive",
        description="Chunking strategy used ('recursive', 'semantic', or 'structural').",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="Model used for dense embeddings."
    )
    embedding_dimensions: int = Field(
        default=384, description="Dimensions of the dense embeddings."
    )

    # ── Enrichment fields (populated by the enrichment pipeline) ──
    summary: str = Field(default="", description="One-sentence summary of the chunk.")
    entities: list[str] = Field(
        default_factory=list,
        description="Named entities extracted from the chunk.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Semantic keywords for retrieval boost.",
    )
    parent_context: str = Field(
        default="",
        description="Section/heading hierarchy this chunk belongs to.",
    )
    prev_summary: str = Field(
        default="",
        description="Summary of the previous chunk for continuity.",
    )
    next_summary: str = Field(
        default="",
        description="Summary of the next chunk for continuity.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Atomicity score — how self-contained this chunk is (0–1).",
    )

    # ── Content Type & Structure ──
    content_type: str = Field(
        default="text",
        description="Type of chunk content: 'text', 'table', or 'figure'.",
    )
    table: TableData | None = Field(
        default=None,
        description="Structured table data if content_type is 'table'.",
    )
    figures: list[FigureRef] = Field(
        default_factory=list,
        description="Figures or images associated with this chunk.",
    )

    # ── Document Graph Navigation & Hierarchy ──
    prev_id: str | None = Field(default=None, description="ID of the preceding chunk.")
    next_id: str | None = Field(default=None, description="ID of the following chunk.")
    parent_id: str | None = Field(
        default=None, description="ID of the parent chunk/section, if any."
    )
    children_ids: list[str] = Field(
        default_factory=list,
        description="IDs of child chunks contained within this chunk/section.",
    )
    section_id: str | None = Field(
        default=None,
        description="Unique section identifier grouping chunks in the same heading branch.",
    )
    depth: int = Field(
        default=0,
        ge=0,
        description="Nesting depth level of the chunk in the document structure.",
    )
    relationships: list[ChunkRelationship] = Field(
        default_factory=list,
        description="Graph edges connecting to related chunks.",
    )

    # ── Provenance ──
    metadata: ChunkMetadata = Field(..., description="Source and sizing metadata.")


class ScoredChunk(BaseModel):
    """A SmartChunk paired with retrieval relevance scores."""

    chunk: SmartChunk = Field(..., description="The underlying SmartChunk.")
    score: float = Field(default=0.0, description="Final combined relevance score.")
    dense_score: float = Field(default=0.0, description="Dense vector similarity score.")
    bm25_score: float = Field(default=0.0, description="Lexical BM25 score.")
    rerank_score: float | None = Field(
        default=None, description="Score assigned by reranker, if applied."
    )


# ── Configuration Models ───────────────────────────────────────────────────────


class ChunkerConfig(BaseModel):
    """Configuration for the chunking stage."""

    strategy: ChunkStrategy = Field(
        default=ChunkStrategy.RECURSIVE,
        description="Which chunking strategy to use.",
    )
    chunk_size: int = Field(
        default=512,
        gt=0,
        description="Target chunk size in tokens.",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Overlap between consecutive chunks in tokens.",
    )
    similarity_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for semantic chunking.",
    )


class EnrichmentConfig(BaseModel):
    """Configuration for the LLM enrichment stage."""

    model: str = Field(
        default="gpt-4o-mini",
        description="LiteLLM model string (e.g. 'gpt-4o-mini', 'ollama/llama3').",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature.",
    )
    enrichments: list[EnrichmentField] = Field(
        default_factory=lambda: list(EnrichmentField),
        description="Which enrichment fields to populate.",
    )
    batch_size: int = Field(
        default=10,
        gt=0,
        description="Number of chunks to enrich in a single batch.",
    )
    max_concurrency: int = Field(
        default=5,
        gt=0,
        description="Max concurrent LLM calls.",
    )
    max_retries: int = Field(
        default=3,
        gt=0,
        description="Max retries per LLM call on failure.",
    )


class RetrievalConfig(BaseModel):
    """Configuration for hybrid retrieval and graph expansion."""

    vector_weight: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Weight for dense vector search."
    )
    bm25_weight: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Weight for BM25 keyword search."
    )
    expand_parents: bool = Field(
        default=False, description="Expand retrieval to include parent sections."
    )
    expand_neighbors: int = Field(
        default=0, ge=0, description="Number of adjacent neighbor chunks (prev/next) to retrieve."
    )
    reranker_model: str | None = Field(
        default=None, description="Cross-encoder model string for reranking stage."
    )
    reranker_top_k: int = Field(
        default=5, gt=0, description="Number of candidates to retain after reranking."
    )


class PipelineConfig(BaseModel):
    """Top-level configuration combining all stages."""

    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    enrich: bool = Field(
        default=True,
        description="Whether to run LLM enrichment at all.",
    )
