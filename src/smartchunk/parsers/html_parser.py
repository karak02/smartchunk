"""HTML parser — extracts structured text, heading hierarchies, tables,
and image references from HTML files.

Requires the ``html`` extra::

    pip install smartchunk[html]
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from smartchunk.models import DocumentSection, FigureRef, TableData
from smartchunk.parsers.base import BaseParser


class HtmlParser(BaseParser):
    """Parses HTML documents into structured sections."""

    supported_extensions = [".html", ".htm"]

    def parse(self, filepath: str | Path) -> list[DocumentSection]:
        filepath = Path(filepath)
        text = filepath.read_text(encoding="utf-8", errors="replace")
        return self.parse_text(text, source=filepath.name)

    def parse_text(self, text: str, source: str = "<string>") -> list[DocumentSection]:
        try:
            from bs4 import BeautifulSoup  # type: ignore[import-untyped]

            soup = BeautifulSoup(text, "html.parser")
            return self._parse_with_bs4(soup, source)
        except ImportError:
            return self._parse_fallback_regex(text, source)

    def _parse_with_bs4(self, soup: Any, source: str) -> list[DocumentSection]:
        sections: list[DocumentSection] = []
        heading_stack: list[tuple[int, str]] = []

        # Find body or main element
        root = soup.find("body") or soup

        for elem in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "table", "img"]):
            tag_name = elem.name.lower()

            if tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(tag_name[1])
                h_text = elem.get_text(strip=True)
                if not h_text:
                    continue

                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, h_text))

            elif tag_name == "p":
                p_text = elem.get_text(strip=True)
                if not p_text:
                    continue

                parent_headings = [h[1] for h in heading_stack]
                heading = heading_stack[-1][1] if heading_stack else None
                level = heading_stack[-1][0] if heading_stack else 0

                sections.append(
                    DocumentSection(
                        text=p_text,
                        heading=heading,
                        level=level,
                        parent_headings=parent_headings,
                        metadata={"source": source},
                    )
                )

            elif tag_name == "table":
                table_data = self._extract_bs4_table(elem)
                if table_data:
                    parent_headings = [h[1] for h in heading_stack]
                    heading = heading_stack[-1][1] if heading_stack else None
                    level = heading_stack[-1][0] if heading_stack else 0

                    sections.append(
                        DocumentSection(
                            text=table_data.to_markdown(),
                            heading=heading,
                            level=level,
                            parent_headings=parent_headings,
                            content_type="table",
                            table=table_data,
                            metadata={"source": source},
                        )
                    )

            elif tag_name == "img":
                src = elem.get("src", "")
                alt = elem.get("alt", "") or elem.get("title", "")
                if src or alt:
                    parent_headings = [h[1] for h in heading_stack]
                    fig = FigureRef(image_path=src, caption=alt)
                    sections.append(
                        DocumentSection(
                            text=f"Figure: {alt}" if alt else f"Image: {src}",
                            heading=heading_stack[-1][1] if heading_stack else None,
                            level=heading_stack[-1][0] if heading_stack else 0,
                            parent_headings=parent_headings,
                            content_type="figure",
                            figures=[fig],
                            metadata={"source": source},
                        )
                    )

        return sections

    def _extract_bs4_table(self, table_elem: Any) -> TableData | None:
        """Extract headers and rows from a BeautifulSoup table tag."""
        caption_tag = table_elem.find("caption")
        caption = caption_tag.get_text(strip=True) if caption_tag else ""

        headers: list[str] = []
        for th in table_elem.find_all("th"):
            headers.append(th.get_text(strip=True))

        rows: list[list[str]] = []
        for tr in table_elem.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if cells:
                rows.append(cells)

        if not headers and not rows:
            return None

        return TableData(headers=headers, rows=rows, caption=caption)

    def _parse_fallback_regex(self, text: str, source: str) -> list[DocumentSection]:
        """Regex fallback if BeautifulSoup is not installed."""
        clean = re.sub(r"<[^>]+>", " ", text)
        clean = " ".join(clean.split())
        return [DocumentSection(text=clean, metadata={"source": source})]
