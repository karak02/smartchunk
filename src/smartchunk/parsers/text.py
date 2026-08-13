"""Plain text parser — paragraph-level splitting with whitespace normalization."""

from __future__ import annotations

import re
from pathlib import Path

from smartchunk.models import DocumentSection
from smartchunk.parsers.base import BaseParser


class TextParser(BaseParser):
    """Parses plain text files into paragraph-level sections.

    Splits on double-newlines (blank lines) and normalises whitespace.
    """

    supported_extensions = [".txt", ".text", ".log"]

    def parse(self, filepath: str | Path) -> list[DocumentSection]:
        filepath = Path(filepath)
        text = filepath.read_text(encoding="utf-8")
        return self.parse_text(text, source=filepath.name)

    def parse_text(self, text: str, source: str = "<string>") -> list[DocumentSection]:
        # Normalise line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Split on one-or-more blank lines
        raw_paragraphs = re.split(r"\n{2,}", text)

        sections: list[DocumentSection] = []
        for para in raw_paragraphs:
            # Collapse internal whitespace runs into single spaces
            cleaned = " ".join(para.split())
            if not cleaned:
                continue
            sections.append(
                DocumentSection(
                    text=cleaned,
                    metadata={"source": source},
                )
            )

        return sections
