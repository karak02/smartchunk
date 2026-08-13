"""XML parser — extracts text from XML files.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from smartchunk.models import DocumentSection
from smartchunk.parsers.base import BaseParser


class XmlParser(BaseParser):
    """Parses XML (.xml) documents."""

    supported_extensions = [".xml"]

    def parse(self, filepath: str | Path) -> list[DocumentSection]:
        filepath = Path(filepath)

        tree = ET.parse(filepath)
        root = tree.getroot()

        # Sequentially collect all text from the XML nodes
        text_parts: list[str] = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                text_parts.append(elem.text.strip())
            if elem.tail and elem.tail.strip():
                text_parts.append(elem.tail.strip())

        cleaned_text = "\n".join(text_parts)

        return [
            DocumentSection(
                text=cleaned_text,
                heading=filepath.stem,
                level=1,
                parent_headings=[],
                metadata={"source": filepath.name},
            )
        ]
