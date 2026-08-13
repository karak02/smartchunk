"""DOCX parser — extracts structured sections, headings, and tables from Word documents.

Requires the ``docx`` extra::

    pip install smartchunk[docx]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from smartchunk.models import DocumentSection, TableData
from smartchunk.parsers.base import BaseParser


class DocxParser(BaseParser):
    """Parses Microsoft Word (.docx) documents.

    Preserves heading levels, paragraph text, and table structures.
    """

    supported_extensions = [".docx"]

    def parse(self, filepath: str | Path) -> list[DocumentSection]:
        filepath = Path(filepath)

        try:
            import docx  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "DOCX parsing requires python-docx. Install with: pip install smartchunk[docx]"
            ) from exc

        doc = docx.Document(str(filepath))
        sections: list[DocumentSection] = []

        heading_stack: list[tuple[int, str]] = []

        for element in doc.element.body:
            # Check for Paragraphs
            if element.tag.endswith("p"):
                p = docx.text.paragraph.Paragraph(element, doc)
                text = p.text.strip()
                if not text:
                    continue

                style_name = p.style.name.lower() if p.style else ""
                if style_name.startswith("heading"):
                    try:
                        level = int(style_name.replace("heading", "").strip())
                    except ValueError:
                        level = 1

                    while heading_stack and heading_stack[-1][0] >= level:
                        heading_stack.pop()
                    heading_stack.append((level, text))
                else:
                    parent_headings = [h[1] for h in heading_stack]
                    heading = heading_stack[-1][1] if heading_stack else None
                    level = heading_stack[-1][0] if heading_stack else 0

                    sections.append(
                        DocumentSection(
                            text=text,
                            heading=heading,
                            level=level,
                            parent_headings=parent_headings,
                            metadata={"source": filepath.name},
                        )
                    )

            # Check for Tables
            elif element.tag.endswith("tbl"):
                t = docx.table.Table(element, doc)
                table_data = self._extract_table_data(t)
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
                            metadata={"source": filepath.name},
                        )
                    )

        return sections

    def _extract_table_data(self, table: Any) -> TableData | None:
        """Convert a python-docx Table object into TableData."""
        rows_data: list[list[str]] = []
        for row in table.rows:
            row_cells = [cell.text.strip() for cell in row.cells]
            if any(row_cells):
                rows_data.append(row_cells)

        if not rows_data:
            return None

        headers = rows_data[0]
        data_rows = rows_data[1:] if len(rows_data) > 1 else []

        return TableData(headers=headers, rows=data_rows)
