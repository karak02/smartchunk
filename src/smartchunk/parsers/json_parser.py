"""JSON parser — extracts structured text from JSON files."""

from __future__ import annotations

import json
from pathlib import Path

from smartchunk.models import DocumentSection
from smartchunk.parsers.base import BaseParser


class JsonParser(BaseParser):
    """Parses JSON (.json) documents."""

    supported_extensions = [".json"]

    def parse(self, filepath: str | Path) -> list[DocumentSection]:
        filepath = Path(filepath)

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        # Prettify the JSON
        formatted_json = json.dumps(data, indent=2)

        return [
            DocumentSection(
                text=formatted_json,
                heading=filepath.stem,
                level=1,
                parent_headings=[],
                metadata={"source": filepath.name},
            )
        ]
