"""Shared test fixtures for SmartChunk tests."""

from __future__ import annotations

import pytest

from smartchunk.models import (
    ChunkerConfig,
    ChunkMetadata,
    ChunkStrategy,
    DocumentSection,
    PipelineConfig,
    SmartChunk,
)


@pytest.fixture
def sample_text() -> str:
    """A multi-paragraph sample text for testing."""
    return (
        "The board of directors met on March 15, 2026 to discuss the annual budget. "
        "CEO Sarah Chen presented the growth strategy for the APAC region, "
        "highlighting a projected revenue increase of 23% year-over-year.\n\n"
        "The board approved a $50M expansion plan targeting Southeast Asian markets. "
        "This capital expenditure will be funded through a combination of retained earnings "
        "and a new credit facility arranged by Goldman Sachs.\n\n"
        "CFO Michael Torres outlined the risk factors associated with the expansion. "
        "Currency fluctuation in emerging markets remains the primary concern, "
        "with potential impact on Q3 and Q4 2026 earnings.\n\n"
        "The meeting concluded with a vote on the new ESG policy framework. "
        "All board members voted in favour, making GreenTech Corp one of the first "
        "in the industry to adopt comprehensive sustainability targets."
    )


@pytest.fixture
def sample_markdown() -> str:
    """A sample markdown document with headings."""
    return (
        "# Annual Report 2026\n\n"
        "This is the executive summary of our annual report.\n\n"
        "## Financial Strategy\n\n"
        "### Capital Allocation\n\n"
        "The board approved a $50M expansion into Southeast Asian markets. "
        "This represents our largest single investment to date.\n\n"
        "### Revenue Projections\n\n"
        "We project 23% year-over-year revenue growth driven by APAC expansion.\n\n"
        "## Risk Factors\n\n"
        "Currency fluctuation in emerging markets remains the primary concern. "
        "We have hedged 60% of our projected FX exposure through Q4 2026.\n\n"
        "## ESG Initiatives\n\n"
        "GreenTech Corp has adopted a comprehensive sustainability framework "
        "targeting net-zero emissions by 2030."
    )


@pytest.fixture
def sample_sections() -> list[DocumentSection]:
    """Pre-parsed document sections for chunker tests."""
    return [
        DocumentSection(
            text=(
                "The board of directors met on March 15, 2026 to discuss the annual budget. "
                "CEO Sarah Chen presented the growth strategy for the APAC region."
            ),
            heading="Board Meeting",
            level=2,
            parent_headings=["Annual Report"],
            metadata={"source": "test.md"},
        ),
        DocumentSection(
            text=(
                "The board approved a $50M expansion plan targeting Southeast Asian markets. "
                "This capital expenditure will be funded through retained earnings."
            ),
            heading="Capital Allocation",
            level=3,
            parent_headings=["Annual Report", "Board Meeting"],
            metadata={"source": "test.md"},
        ),
        DocumentSection(
            text=(
                "CFO Michael Torres outlined the risk factors. "
                "Currency fluctuation in emerging markets remains the primary concern."
            ),
            heading="Risk Factors",
            level=2,
            parent_headings=["Annual Report"],
            metadata={"source": "test.md"},
        ),
    ]


@pytest.fixture
def sample_smart_chunks() -> list[SmartChunk]:
    """Pre-built SmartChunks for exporter tests."""
    return [
        SmartChunk(
            id="chunk_test_001",
            text="The board approved a $50M expansion plan.",
            summary="Board approves major capital expenditure",
            entities=["Board of Directors", "$50M"],
            keywords=["budget", "expansion", "capital"],
            parent_context="Annual Report → Financial Strategy",
            confidence=0.92,
            metadata=ChunkMetadata(
                source="test.pdf",
                page=1,
                chunk_index=0,
                total_chunks=3,
                char_count=42,
                token_count=11,
            ),
        ),
        SmartChunk(
            id="chunk_test_002",
            text="Revenue growth projected at 23% YoY.",
            summary="Revenue projections show strong growth",
            entities=["23%"],
            keywords=["revenue", "growth", "projections"],
            parent_context="Annual Report → Revenue",
            confidence=0.88,
            metadata=ChunkMetadata(
                source="test.pdf",
                page=2,
                chunk_index=1,
                total_chunks=3,
                char_count=37,
                token_count=9,
            ),
        ),
        SmartChunk(
            id="chunk_test_003",
            text="Currency risk is the primary concern for Q3-Q4 2026.",
            summary="FX risk highlighted as main concern",
            entities=["Q3 2026", "Q4 2026"],
            keywords=["currency", "risk", "emerging markets"],
            parent_context="Annual Report → Risk Factors",
            confidence=0.95,
            metadata=ChunkMetadata(
                source="test.pdf",
                page=3,
                chunk_index=2,
                total_chunks=3,
                char_count=52,
                token_count=13,
            ),
        ),
    ]


@pytest.fixture
def recursive_config() -> ChunkerConfig:
    """Chunker config with recursive strategy."""
    return ChunkerConfig(
        strategy=ChunkStrategy.RECURSIVE,
        chunk_size=100,
        chunk_overlap=10,
    )


@pytest.fixture
def no_enrich_config() -> PipelineConfig:
    """Pipeline config with enrichment disabled."""
    return PipelineConfig(enrich=False)


@pytest.fixture(autouse=True)
def mock_llm_cache(tmp_path, monkeypatch):
    """Bypass persistent cache for all tests to prevent test pollution."""
    temp_cache_file = tmp_path / ".smartchunk_cache.json"
    from smartchunk.enrichment.llm import LLMEnricher

    original_init = LLMEnricher.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._cache_file = str(temp_cache_file)
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    monkeypatch.setattr(LLMEnricher, "__init__", patched_init)
