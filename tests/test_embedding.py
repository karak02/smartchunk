"""Tests for ContextualEmbeddingBuilder."""

from smartchunk.embedding import ContextualEmbeddingBuilder
from smartchunk.models import ChunkMetadata, SmartChunk


def test_contextual_embedding_builder():
    chunk = SmartChunk(
        text="The board approved $50M expansion.",
        summary="Board approves capital expenditure.",
        parent_context="Financial Strategy → Capital Allocation",
        metadata=ChunkMetadata(source="annual_report.pdf", chunk_index=0, total_chunks=1, char_count=35, token_count=7),
    )

    builder = ContextualEmbeddingBuilder()
    builder.build([chunk], document_title="Annual Report 2026")

    ctx = chunk.contextual_text
    assert "Document: Annual Report 2026" in ctx
    assert "Section: Financial Strategy → Capital Allocation" in ctx
    assert "Summary: Board approves capital expenditure." in ctx
    assert "The board approved $50M expansion." in ctx
