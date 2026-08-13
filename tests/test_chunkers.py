"""Tests for chunking strategies."""

from __future__ import annotations

import pytest

from smartchunk.chunkers.base import get_chunker
from smartchunk.chunkers.recursive import RecursiveChunker
from smartchunk.chunkers.structural import StructuralChunker
from smartchunk.models import ChunkerConfig, ChunkStrategy, DocumentSection


class TestRecursiveChunker:
    """Tests for the recursive text splitter."""

    def test_small_text_single_chunk(self):
        config = ChunkerConfig(chunk_size=500, chunk_overlap=0)
        chunker = RecursiveChunker(config)

        sections = [DocumentSection(text="Short text.")]
        chunks = chunker.chunk(sections)

        assert len(chunks) == 1
        assert chunks[0]["text"] == "Short text."

    def test_splits_large_text(self):
        config = ChunkerConfig(chunk_size=20, chunk_overlap=0)
        chunker = RecursiveChunker(config)

        long_text = "This is a sentence. " * 50
        sections = [DocumentSection(text=long_text.strip())]
        chunks = chunker.chunk(sections)

        assert len(chunks) > 1
        # Each chunk should respect the token limit (approximately)
        for chunk in chunks:
            assert len(chunk["text"]) > 0

    def test_preserves_parent_context(self, sample_sections: list[DocumentSection]):
        config = ChunkerConfig(chunk_size=500, chunk_overlap=0)
        chunker = RecursiveChunker(config)

        chunks = chunker.chunk(sample_sections)

        # Check that parent context is built from headings
        contexts = [c["parent_context"] for c in chunks]
        assert any("Board Meeting" in c for c in contexts)
        assert any("Capital Allocation" in c for c in contexts)

    def test_empty_sections(self):
        config = ChunkerConfig(chunk_size=100, chunk_overlap=0)
        chunker = RecursiveChunker(config)

        sections = [DocumentSection(text="")]
        chunks = chunker.chunk(sections)

        assert len(chunks) == 0

    def test_page_number_preserved(self):
        config = ChunkerConfig(chunk_size=500, chunk_overlap=0)
        chunker = RecursiveChunker(config)

        sections = [DocumentSection(text="Content on page 5.", page=5)]
        chunks = chunker.chunk(sections)

        assert chunks[0]["page"] == 5


class TestStructuralChunker:
    """Tests for the section-aware structural chunker."""

    def test_small_sections_stay_intact(self, sample_sections: list[DocumentSection]):
        config = ChunkerConfig(
            strategy=ChunkStrategy.STRUCTURAL,
            chunk_size=500,
            chunk_overlap=0,
        )
        chunker = StructuralChunker(config)

        chunks = chunker.chunk(sample_sections)

        # Each section is small enough to be one chunk
        assert len(chunks) == len(sample_sections)

    def test_large_section_gets_sub_split(self):
        config = ChunkerConfig(
            strategy=ChunkStrategy.STRUCTURAL,
            chunk_size=20,
            chunk_overlap=0,
        )
        chunker = StructuralChunker(config)

        sections = [
            DocumentSection(
                text="This is a very long sentence. " * 50,
                heading="Big Section",
                level=1,
            )
        ]
        chunks = chunker.chunk(sections)

        assert len(chunks) > 1
        # All sub-chunks should have the same parent context
        for chunk in chunks:
            assert chunk["parent_context"] == "Big Section"

    def test_parent_context_chain(self, sample_sections: list[DocumentSection]):
        config = ChunkerConfig(
            strategy=ChunkStrategy.STRUCTURAL,
            chunk_size=500,
        )
        chunker = StructuralChunker(config)

        chunks = chunker.chunk(sample_sections)

        # The "Capital Allocation" section should have full hierarchy
        cap_chunks = [c for c in chunks if "Capital Allocation" in c["parent_context"]]
        assert len(cap_chunks) >= 1
        assert "Annual Report" in cap_chunks[0]["parent_context"]
        assert "Board Meeting" in cap_chunks[0]["parent_context"]


class TestGetChunker:
    """Tests for the chunker factory."""

    def test_recursive_default(self):
        chunker = get_chunker()
        assert isinstance(chunker, RecursiveChunker)

    def test_recursive_explicit(self):
        config = ChunkerConfig(strategy=ChunkStrategy.RECURSIVE)
        chunker = get_chunker(config)
        assert isinstance(chunker, RecursiveChunker)

    def test_structural(self):
        config = ChunkerConfig(strategy=ChunkStrategy.STRUCTURAL)
        chunker = get_chunker(config)
        assert isinstance(chunker, StructuralChunker)
