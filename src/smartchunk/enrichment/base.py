"""Abstract base enricher."""

from __future__ import annotations

import abc

from smartchunk.models import EnrichmentConfig, SmartChunk


class BaseEnricher(abc.ABC):
    """Base class for chunk enrichment strategies."""

    def __init__(self, config: EnrichmentConfig | None = None) -> None:
        self.config = config or EnrichmentConfig()

    @abc.abstractmethod
    def enrich(self, chunks: list[SmartChunk]) -> list[SmartChunk]:
        """Enrich a list of SmartChunks with metadata.

        This mutates the chunks in-place AND returns them for chaining.

        Parameters
        ----------
        chunks:
            List of SmartChunks with at least ``text`` populated.

        Returns
        -------
        list[SmartChunk]
            The same list, with enrichment fields populated.
        """
