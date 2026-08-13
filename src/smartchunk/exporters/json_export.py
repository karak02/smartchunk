"""JSON / JSONL file exporter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smartchunk.exporters.base import BaseExporter
from smartchunk.models import SmartChunk


class JsonExporter(BaseExporter):
    """Exports SmartChunks to JSON or JSONL files."""

    def export(self, chunks: list[SmartChunk], **kwargs: object) -> None:
        """Not used directly — use ``to_json`` or ``to_jsonl`` instead."""
        filepath = kwargs.get("filepath")
        if filepath:
            self.to_json(chunks, str(filepath))

    @staticmethod
    def to_json(chunks: list[SmartChunk], filepath: str | Path) -> None:
        """Write chunks as a pretty-printed JSON array.

        Parameters
        ----------
        chunks:
            List of SmartChunks to export.
        filepath:
            Output file path.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = [chunk.model_dump(mode="json") for chunk in chunks]
        filepath.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def to_jsonl(chunks: list[SmartChunk], filepath: str | Path) -> None:
        """Write chunks as newline-delimited JSON (JSONL).

        Parameters
        ----------
        chunks:
            List of SmartChunks to export.
        filepath:
            Output file path.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with filepath.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                line = json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False)
                f.write(line + "\n")

    @staticmethod
    def to_dict(chunks: list[SmartChunk]) -> list[dict[str, Any]]:
        """Convert chunks to a list of plain dictionaries.

        Parameters
        ----------
        chunks:
            List of SmartChunks to convert.

        Returns
        -------
        list[dict]
            List of chunk data as dicts.
        """
        return [chunk.model_dump(mode="json") for chunk in chunks]
