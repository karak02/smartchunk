"""Abstract base exporter."""

from __future__ import annotations

import abc

from smartchunk.models import SmartChunk


class BaseExporter(abc.ABC):
    """Base class for all chunk exporters."""

    @abc.abstractmethod
    def export(self, chunks: list[SmartChunk], **kwargs: object) -> None:
        """Export chunks to the target destination.

        Parameters
        ----------
        chunks:
            List of enriched SmartChunks to export.
        **kwargs:
            Exporter-specific options.
        """
