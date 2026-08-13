"""Tests for the SmartChunker pipeline orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smartchunk import PipelineConfig, SmartChunk, SmartChunker


class TestSmartChunkerInit:
    """Tests for SmartChunker initialization."""

    def test_default_init(self):
        chunker = SmartChunker()
        assert chunker.config.enrich is True
        assert chunker.config.chunker.chunk_size == 512

    def test_custom_params(self):
        chunker = SmartChunker(
            model="claude-3-haiku-20240307",
            chunk_size=256,
            strategy="structural",
            enrich=False,
        )
        assert chunker.config.enrichment.model == "claude-3-haiku-20240307"
        assert chunker.config.chunker.chunk_size == 256
        assert chunker.config.chunker.strategy.value == "structural"
        assert chunker.config.enrich is False

    def test_custom_config_object(self):
        config = PipelineConfig(enrich=False)
        chunker = SmartChunker(config=config)
        assert chunker.config.enrich is False


class TestSmartChunkerProcess:
    """Tests for document processing (no LLM calls)."""

    def test_process_text_file(self, tmp_path: Path, sample_text: str):
        file = tmp_path / "test.txt"
        file.write_text(sample_text, encoding="utf-8")

        chunker = SmartChunker(enrich=False)
        chunks = chunker.process(file)

        assert len(chunks) > 0
        assert all(isinstance(c, SmartChunk) for c in chunks)
        assert all(c.metadata.source == "test.txt" for c in chunks)
        assert chunks[0].metadata.chunk_index == 0
        assert chunks[0].metadata.total_chunks == len(chunks)
        assert chunks[0].metadata.char_count > 0
        assert chunks[0].metadata.token_count > 0

    def test_process_markdown_file(self, tmp_path: Path, sample_markdown: str):
        file = tmp_path / "test.md"
        file.write_text(sample_markdown, encoding="utf-8")

        chunker = SmartChunker(enrich=False, strategy="structural")
        chunks = chunker.process(file)

        assert len(chunks) > 0
        # Should have parent context from headings
        contexts = [c.parent_context for c in chunks]
        assert any("Financial Strategy" in c for c in contexts if c)

    def test_process_text_string(self, sample_text: str):
        chunker = SmartChunker(enrich=False)
        chunks = chunker.process_text(sample_text, source="my_doc")

        assert len(chunks) > 0
        assert all(c.metadata.source == "my_doc" for c in chunks)

    def test_process_file_not_found(self):
        chunker = SmartChunker(enrich=False)
        with pytest.raises(FileNotFoundError):
            chunker.process("nonexistent_file.txt")


class TestSmartChunkerExport:
    """Tests for export shortcuts."""

    def test_to_json_shortcut(self, tmp_path: Path, sample_text: str):
        file = tmp_path / "test.txt"
        file.write_text(sample_text, encoding="utf-8")

        chunker = SmartChunker(enrich=False)
        chunks = chunker.process(file)

        output = tmp_path / "output.json"
        SmartChunker.to_json(chunks, output)

        data = json.loads(output.read_text(encoding="utf-8"))
        assert len(data) == len(chunks)

    def test_to_jsonl_shortcut(self, tmp_path: Path, sample_text: str):
        file = tmp_path / "test.txt"
        file.write_text(sample_text, encoding="utf-8")

        chunker = SmartChunker(enrich=False)
        chunks = chunker.process(file)

        output = tmp_path / "output.jsonl"
        SmartChunker.to_jsonl(chunks, output)

        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == len(chunks)

    def test_to_dict_shortcut(self, sample_text: str):
        chunker = SmartChunker(enrich=False)
        chunks = chunker.process_text(sample_text)

        dicts = SmartChunker.to_dict(chunks)
        assert isinstance(dicts, list)
        assert all(isinstance(d, dict) for d in dicts)

    def test_usage_stats_no_enrichment(self):
        chunker = SmartChunker(enrich=False)
        stats = chunker.usage_stats
        assert stats["total_tokens"] == 0


class TestSmartChunkerWithEnrichment:
    """Tests with mocked LLM enrichment."""

    @patch("smartchunk.enrichment.llm.acompletion")
    def test_full_pipeline_with_enrichment(
        self, mock_acompletion: AsyncMock, tmp_path: Path, sample_text: str
    ):
        mock_data = {
            "summary": "Test summary",
            "entities": ["Entity1"],
            "keywords": ["key1", "key2"],
            "confidence": 0.85,
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(mock_data)
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_acompletion.return_value = mock_response

        file = tmp_path / "test.txt"
        file.write_text(sample_text, encoding="utf-8")

        chunker = SmartChunker(model="gpt-4o-mini", enrich=True, max_concurrency=1)
        chunks = chunker.process(file)

        assert len(chunks) > 0
        assert all(c.summary == "Test summary" for c in chunks)
        assert all(c.entities == ["Entity1"] for c in chunks)
        assert all(c.confidence == pytest.approx(0.85) for c in chunks)

        # Neighbor linking
        if len(chunks) > 1:
            assert chunks[1].prev_summary != ""
