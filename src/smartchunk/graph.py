"""Chunk graph builder — constructs navigational and structural relationships between chunks.

Connects chunks sequentially (prev_id/next_id), hierarchically (sections/depth),
and semantically (shared entities).
"""

from __future__ import annotations

import hashlib
import logging

from smartchunk.models import ChunkRelationship, RelationshipType, SmartChunk

logger = logging.getLogger("smartchunk.graph")


class ChunkGraphBuilder:
    """Builds a graph structure across an ordered list of SmartChunks.

    Establishes:
    - Sequential links (prev_id, next_id, PRECEDES, FOLLOWS)
    - Section grouping (section_id, depth, SAME_SECTION)
    - Entity links (SHARED_ENTITY)
    """

    def build(self, chunks: list[SmartChunk]) -> list[SmartChunk]:
        """Build graph links for a list of chunks.

        Parameters
        ----------
        chunks:
            Ordered list of SmartChunk objects.

        Returns
        -------
        list[SmartChunk]
            The enriched list of chunks with graph attributes populated.
        """
        if not chunks:
            return chunks

        self._link_sequential(chunks)
        self._link_sections(chunks)
        return chunks

    def _link_sequential(self, chunks: list[SmartChunk]) -> None:
        """Establish prev_id / next_id and PRECEDES / FOLLOWS relationships."""
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            # Preceding chunk
            if i > 0:
                prev_chunk = chunks[i - 1]
                chunk.prev_id = prev_chunk.id
                chunk.relationships.append(
                    ChunkRelationship(
                        target_id=prev_chunk.id,
                        relation=RelationshipType.FOLLOWS,
                    )
                )

            # Following chunk
            if i < total - 1:
                next_chunk = chunks[i + 1]
                chunk.next_id = next_chunk.id
                chunk.relationships.append(
                    ChunkRelationship(
                        target_id=next_chunk.id,
                        relation=RelationshipType.PRECEDES,
                    )
                )

    def _link_sections(self, chunks: list[SmartChunk]) -> None:
        """Group chunks into sections based on parent_context hierarchy."""
        section_map: dict[str, list[SmartChunk]] = {}

        for chunk in chunks:
            context = chunk.parent_context.strip()
            if context:
                # Generate deterministic section ID from parent_context
                sec_hash = hashlib.md5(context.encode("utf-8")).hexdigest()[:10]
                sec_id = f"sec_{sec_hash}"
                chunk.section_id = sec_id
                
                # Calculate depth from hierarchy length ("Section A → Sub B" -> depth 2)
                chunk.depth = len([p for p in context.split("→") if p.strip()])
                
                section_map.setdefault(sec_id, []).append(chunk)
            else:
                chunk.depth = 0

        # Link chunks in the same section
        for sec_id, sec_chunks in section_map.items():
            if len(sec_chunks) <= 1:
                continue

            sec_chunk_ids = [c.id for c in sec_chunks]
            for chunk in sec_chunks:
                for target_id in sec_chunk_ids:
                    if target_id != chunk.id:
                        # Avoid duplicates
                        if not any(
                            r.target_id == target_id and r.relation == RelationshipType.SAME_SECTION
                            for r in chunk.relationships
                        ):
                            chunk.relationships.append(
                                ChunkRelationship(
                                    target_id=target_id,
                                    relation=RelationshipType.SAME_SECTION,
                                )
                            )

    def link_entities(self, chunks: list[SmartChunk]) -> None:
        """Post-enrichment step: link chunks that share named entities."""
        entity_to_chunks: dict[str, list[str]] = {}

        for chunk in chunks:
            for entity in chunk.entities:
                entity_clean = entity.strip().lower()
                if entity_clean:
                    entity_to_chunks.setdefault(entity_clean, []).append(chunk.id)

        chunk_by_id = {c.id: c for c in chunks}

        for entity, chunk_ids in entity_to_chunks.items():
            if len(chunk_ids) <= 1:
                continue

            for cid in chunk_ids:
                chunk = chunk_by_id.get(cid)
                if not chunk:
                    continue

                for target_id in chunk_ids:
                    if target_id != cid:
                        if not any(
                            r.target_id == target_id and r.relation == RelationshipType.SHARED_ENTITY
                            for r in chunk.relationships
                        ):
                            chunk.relationships.append(
                                ChunkRelationship(
                                    target_id=target_id,
                                    relation=RelationshipType.SHARED_ENTITY,
                                    weight=0.8,
                                )
                            )
