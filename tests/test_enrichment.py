"""Tests for the LLM enrichment engine (with mocked LLM calls)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smartchunk.enrichment.llm import LLMEnricher
from smartchunk.enrichment.prompts import SYSTEM_PROMPT, build_enrichment_prompt
from smartchunk.models import (
    ChunkMetadata,
    EnrichmentConfig,
    EnrichmentField,
    SmartChunk,
)


def _make_chunk(text: str, chunk_index: int = 0) -> SmartChunk:
    """Helper to create a minimal SmartChunk for testing."""
    return SmartChunk(
        text=text,
        metadata=ChunkMetadata(
            source="test.txt",
            chunk_index=chunk_index,
            total_chunks=3,
            char_count=len(text),
            token_count=10,
        ),
    )


def _mock_llm_response(data: dict) -> MagicMock:
    """Create a mock LLM response with structured JSON."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(data)
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    return mock_response


class TestPrompts:
    """Tests for prompt template construction."""

    def test_build_prompt_with_context(self):
        prompt = build_enrichment_prompt(
            chunk_text="The board approved a $50M plan.",
            parent_context="Annual Report → Financial Strategy",
        )
        assert "The board approved a $50M plan." in prompt
        assert "Annual Report → Financial Strategy" in prompt
        assert "No additional context" not in prompt

    def test_build_prompt_without_context(self):
        prompt = build_enrichment_prompt(
            chunk_text="Some text without context.",
            parent_context="",
        )
        assert "Some text without context." in prompt
        assert "No additional context" in prompt

    def test_system_prompt_exists(self):
        assert len(SYSTEM_PROMPT) > 0
        assert "JSON" in SYSTEM_PROMPT


class TestLLMEnricher:
    """Tests for the LLM enrichment engine with mocked calls."""

    @patch("smartchunk.enrichment.llm.acompletion")
    def test_enrich_populates_all_fields(self, mock_acompletion: AsyncMock):
        """Test that enrichment populates summary, entities, keywords, confidence."""
        mock_data = {
            "summary": "Board approves major capital expenditure",
            "entities": ["Board of Directors", "$50M"],
            "keywords": ["budget", "expansion", "capital"],
            "confidence": 0.92,
        }
        mock_acompletion.return_value = _mock_llm_response(mock_data)

        config = EnrichmentConfig(model="gpt-4o-mini", max_concurrency=1)
        enricher = LLMEnricher(config)

        chunks = [_make_chunk("The board approved a $50M expansion plan.")]
        result = enricher.enrich(chunks)

        assert result[0].summary == "Board approves major capital expenditure"
        assert "Board of Directors" in result[0].entities
        assert "$50M" in result[0].entities
        assert "budget" in result[0].keywords
        assert result[0].confidence == pytest.approx(0.92)

    @patch("smartchunk.enrichment.llm.acompletion")
    def test_neighbor_linking(self, mock_acompletion: AsyncMock):
        """Test that prev_summary and next_summary are linked correctly."""
        responses = [
            _mock_llm_response(
                {"summary": f"Summary {i}", "entities": [], "keywords": [], "confidence": 0.9}
            )
            for i in range(3)
        ]
        mock_acompletion.side_effect = responses

        config = EnrichmentConfig(model="gpt-4o-mini", max_concurrency=1)
        enricher = LLMEnricher(config)

        chunks = [_make_chunk(f"Chunk {i}", chunk_index=i) for i in range(3)]
        result = enricher.enrich(chunks)

        # First chunk has no prev_summary
        assert result[0].prev_summary == ""
        assert result[0].next_summary == "Summary 1"

        # Middle chunk has both
        assert result[1].prev_summary == "Summary 0"
        assert result[1].next_summary == "Summary 2"

        # Last chunk has no next_summary
        assert result[2].prev_summary == "Summary 1"
        assert result[2].next_summary == ""

    @patch("smartchunk.enrichment.llm.acompletion")
    def test_usage_stats_tracked(self, mock_acompletion: AsyncMock):
        """Test that token usage is accumulated."""
        mock_acompletion.return_value = _mock_llm_response(
            {"summary": "Test", "entities": [], "keywords": [], "confidence": 0.5}
        )

        config = EnrichmentConfig(model="gpt-4o-mini", max_concurrency=1)
        enricher = LLMEnricher(config)

        chunks = [_make_chunk("Test chunk")]
        enricher.enrich(chunks)

        stats = enricher.usage_stats
        assert stats["prompt_tokens"] > 0
        assert stats["completion_tokens"] > 0
        assert stats["total_tokens"] == stats["prompt_tokens"] + stats["completion_tokens"]

    @patch("smartchunk.enrichment.llm.acompletion")
    def test_selective_enrichment(self, mock_acompletion: AsyncMock):
        """Test that only selected enrichment fields are populated."""
        mock_acompletion.return_value = _mock_llm_response(
            {
                "summary": "Test summary",
                "entities": ["Entity1"],
                "keywords": ["key1"],
                "confidence": 0.8,
            }
        )

        # Only enrich summary and keywords
        config = EnrichmentConfig(
            model="gpt-4o-mini",
            enrichments=[EnrichmentField.SUMMARY, EnrichmentField.KEYWORDS],
            max_concurrency=1,
        )
        enricher = LLMEnricher(config)

        chunks = [_make_chunk("Test chunk")]
        result = enricher.enrich(chunks)

        assert result[0].summary == "Test summary"
        assert result[0].keywords == ["key1"]
        # Entities and confidence should remain default since they weren't in enrichments
        assert result[0].entities == []
        assert result[0].confidence == 0.0

    def test_enrich_empty_list(self):
        """Test that enriching an empty list returns empty."""
        enricher = LLMEnricher()
        result = enricher.enrich([])
        assert result == []

    @patch("smartchunk.enrichment.llm.acompletion")
    def test_confidence_clamped(self, mock_acompletion: AsyncMock):
        """Test that confidence values are clamped to [0, 1]."""
        mock_acompletion.return_value = _mock_llm_response(
            {"summary": "Test", "entities": [], "keywords": [], "confidence": 1.5}
        )

        enricher = LLMEnricher(EnrichmentConfig(max_concurrency=1))
        chunks = [_make_chunk("Test")]
        result = enricher.enrich(chunks)

        assert result[0].confidence == 1.0

    @patch("smartchunk.enrichment.llm.acompletion")
    def test_caching_duplicate_chunks(self, mock_acompletion: AsyncMock):
        """Test that duplicate chunk texts reuse cached values and only call LLM once."""
        mock_data = {
            "summary": "Duplicate summary content",
            "entities": ["entity"],
            "keywords": ["key"],
            "confidence": 0.9,
        }
        mock_acompletion.return_value = _mock_llm_response(mock_data)

        config = EnrichmentConfig(model="gpt-4o-mini", max_concurrency=1)
        enricher = LLMEnricher(config)

        # Same text in both chunks
        chunks = [
            _make_chunk("Duplicate text description.", chunk_index=0),
            _make_chunk("Duplicate text description.", chunk_index=1),
        ]
        result = enricher.enrich(chunks)

        # Asserts
        assert result[0].summary == "Duplicate summary content"
        assert result[1].summary == "Duplicate summary content"

        # Verify LLM was only called once
        assert mock_acompletion.call_count == 1

        # Verify cache stats
        c_stats = enricher.cache_stats
        assert c_stats["cache_hits"] == 1
        assert c_stats["cache_misses"] == 1
        assert c_stats["hit_rate_percent"] == 50.0
        assert c_stats["llm_calls_saved"] == 1
        assert c_stats["cache_entries"] == 1
        assert c_stats["estimated_time_saved_seconds"] > 0.0

    def test_cache_injection_and_export(self):
        """Test cache import (inject_cache) and export (export_cache) APIs."""
        enricher = LLMEnricher()
        dummy_cache = {
            "some_hash": {
                "summary": "Injected summary",
                "entities": ["E1"],
                "keywords": ["K1"],
                "confidence": 0.95,
            }
        }
        enricher.inject_cache(dummy_cache)

        # Export and verify content
        exported = enricher.export_cache()
        assert exported == dummy_cache
        assert enricher.cache_stats["cache_entries"] == 1
