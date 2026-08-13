"""Tests for ChunkGraphBuilder and relationship generation."""

from smartchunk.graph import ChunkGraphBuilder
from smartchunk.models import ChunkMetadata, RelationshipType, SmartChunk


def test_sequential_graph_linking():
    chunks = [
        SmartChunk(
            text="Chunk 1",
            metadata=ChunkMetadata(
                source="doc.txt", chunk_index=0, total_chunks=3, char_count=7, token_count=2
            ),
        ),
        SmartChunk(
            text="Chunk 2",
            metadata=ChunkMetadata(
                source="doc.txt", chunk_index=1, total_chunks=3, char_count=7, token_count=2
            ),
        ),
        SmartChunk(
            text="Chunk 3",
            metadata=ChunkMetadata(
                source="doc.txt", chunk_index=2, total_chunks=3, char_count=7, token_count=2
            ),
        ),
    ]

    builder = ChunkGraphBuilder()
    builder.build(chunks)

    assert chunks[0].prev_id is None
    assert chunks[0].next_id == chunks[1].id

    assert chunks[1].prev_id == chunks[0].id
    assert chunks[1].next_id == chunks[2].id

    assert chunks[2].prev_id == chunks[1].id
    assert chunks[2].next_id is None

    # Check relationship edges
    rel_types_0 = [r.relation for r in chunks[0].relationships]
    assert RelationshipType.PRECEDES in rel_types_0

    rel_types_1 = [r.relation for r in chunks[1].relationships]
    assert RelationshipType.FOLLOWS in rel_types_1
    assert RelationshipType.PRECEDES in rel_types_1


def test_section_and_entity_linking():
    chunks = [
        SmartChunk(
            text="Revenue increased.",
            parent_context="Financials → Q3",
            entities=["Acme Corp", "2026"],
            metadata=ChunkMetadata(
                source="report.md", chunk_index=0, total_chunks=2, char_count=18, token_count=3
            ),
        ),
        SmartChunk(
            text="Expenses were low.",
            parent_context="Financials → Q3",
            entities=["Acme Corp", "USD"],
            metadata=ChunkMetadata(
                source="report.md", chunk_index=1, total_chunks=2, char_count=18, token_count=3
            ),
        ),
    ]

    builder = ChunkGraphBuilder()
    builder.build(chunks)
    builder.link_entities(chunks)

    assert chunks[0].section_id is not None
    assert chunks[0].section_id == chunks[1].section_id
    assert chunks[0].depth == 2

    # Same section relationships
    c0_rel = [r.relation for r in chunks[0].relationships]
    assert RelationshipType.SAME_SECTION in c0_rel
    assert RelationshipType.SHARED_ENTITY in c0_rel
