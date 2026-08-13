"""Abstract base chunker and factory."""

from __future__ import annotations

import abc

from smartchunk.models import ChunkerConfig, ChunkStrategy, DocumentSection


class BaseChunker(abc.ABC):
    """Base class for all chunking strategies.

    Subclasses split a list of :class:`DocumentSection` objects into
    a flat list of text chunks with parent context preserved.
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()

    @abc.abstractmethod
    def chunk(self, sections: list[DocumentSection]) -> list[dict]:
        """Split sections into chunks.

        Parameters
        ----------
        sections:
            Ordered list of document sections from a parser.

        Returns
        -------
        list[dict]
            Each dict has at least:
            - ``text`` (str): The chunk text.
            - ``parent_context`` (str): Heading/section hierarchy.
            - ``page`` (int | None): Page number if available.
        """


def get_chunker(config: ChunkerConfig | None = None) -> BaseChunker:
    """Return a chunker instance based on the configured strategy.

    Parameters
    ----------
    config:
        Chunker configuration. Uses defaults if ``None``.
    """
    config = config or ChunkerConfig()

    if config.strategy == ChunkStrategy.SEMANTIC:
        from smartchunk.chunkers.semantic import SemanticChunker

        return SemanticChunker(config)

    if config.strategy == ChunkStrategy.STRUCTURAL:
        from smartchunk.chunkers.structural import StructuralChunker

        return StructuralChunker(config)

    # Default: recursive
    from smartchunk.chunkers.recursive import RecursiveChunker

    return RecursiveChunker(config)
