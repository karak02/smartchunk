"""EPUB parser — extracts structured sections and text from EPUB ebooks.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from html.parser import HTMLParser

from smartchunk.models import DocumentSection
from smartchunk.parsers.base import BaseParser


class HTMLTextExtractor(HTMLParser):
    """Simple HTML parser to extract clean text from XHTML files."""

    def __init__(self) -> None:
        super().__init__()
        self.text_parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.text_parts.append(text)

    def get_text(self) -> str:
        return " ".join(self.text_parts)


class EpubParser(BaseParser):
    """Parses EPUB (.epub) ebook files."""

    supported_extensions = [".epub"]

    def parse(self, filepath: str | Path) -> list[DocumentSection]:
        filepath = Path(filepath)

        sections: list[DocumentSection] = []

        with zipfile.ZipFile(filepath, "r") as zip_ref:
            # Find all HTML/XHTML content documents
            html_files = [
                name for name in zip_ref.namelist()
                if re.search(r"\.(html|xhtml|htm)$", name, re.IGNORECASE)
            ]
            # Sort files to try to preserve reading order
            html_files.sort()

            for name in html_files:
                with zip_ref.open(name) as f:
                    content = f.read().decode("utf-8", errors="ignore")

                # Extract text using our lightweight HTMLTextExtractor
                extractor = HTMLTextExtractor()
                extractor.feed(content)
                text = extractor.get_text()

                if not text.strip():
                    continue

                section_name = Path(name).stem
                sections.append(
                    DocumentSection(
                        text=text,
                        heading=section_name,
                        level=1,
                        parent_headings=[],
                        metadata={"source": filepath.name, "epub_file": name},
                    )
                )

        return sections
