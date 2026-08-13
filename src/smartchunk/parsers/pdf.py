"""PDF parser — extracts structured text, tables, and figures/images via pymupdf4llm and PyMuPDF.

Requires the ``pdf`` extra::

    pip install smartchunk[pdf]
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from smartchunk.models import DocumentSection, FigureRef, TableData
from smartchunk.parsers.base import BaseParser

logger = logging.getLogger("smartchunk.parsers.pdf")

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#*)?$", re.MULTILINE)


def _extract_tables_from_text(text: str) -> list[dict[str, Any]]:
    """Scan markdown text line-by-line and isolate markdown table blocks."""
    lines = text.splitlines()
    blocks: list[dict[str, Any]] = []
    current_text_lines: list[str] = []
    current_table_lines: list[str] = []

    in_table = False

    for line in lines:
        trimmed = line.strip()
        # Table rows in markdown start and end with '|'
        is_table_row = trimmed.startswith("|") and trimmed.endswith("|")

        if is_table_row:
            if not in_table:
                if current_text_lines:
                    blocks.append({"type": "text", "text": "\n".join(current_text_lines)})
                    current_text_lines = []
                in_table = True
            current_table_lines.append(trimmed)
        else:
            if in_table:
                table_data = _parse_markdown_table(current_table_lines)
                if table_data:
                    blocks.append(
                        {
                            "type": "table",
                            "text": "\n".join(current_table_lines),
                            "table": table_data,
                        }
                    )
                else:
                    blocks.append({"type": "text", "text": "\n".join(current_table_lines)})
                current_table_lines = []
                in_table = False
            current_text_lines.append(line)

    if in_table:
        table_data = _parse_markdown_table(current_table_lines)
        if table_data:
            blocks.append(
                {"type": "table", "text": "\n".join(current_table_lines), "table": table_data}
            )
        else:
            blocks.append({"type": "text", "text": "\n".join(current_table_lines)})
    elif current_text_lines:
        blocks.append({"type": "text", "text": "\n".join(current_text_lines)})

    return blocks


def _parse_markdown_table(table_lines: list[str]) -> TableData | None:
    """Parse a block of markdown table lines into a TableData object."""
    if len(table_lines) < 2:
        return None

    # First row is headers
    header_row = table_lines[0]
    headers = [cell.strip() for cell in header_row.split("|")[1:-1]]

    # Second row is the separator line (e.g. |---|), skip it
    rows: list[list[str]] = []
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if any(cells):
            rows.append(cells)

    return TableData(headers=headers, rows=rows)


class PdfParser(BaseParser):
    """Parses PDF files using ``pymupdf4llm`` and PyMuPDF layout analysis.

    Extracts text, table grids, and image/figure locations.
    """

    supported_extensions = [".pdf"]

    def parse(self, filepath: str | Path) -> list[DocumentSection]:
        filepath = Path(filepath)

        try:
            import pymupdf4llm  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PDF parsing requires pymupdf4llm. Install it with: pip install smartchunk[pdf]"
            ) from exc

        # Open doc via PyMuPDF (fitz) for layout/figure extraction
        doc = None
        try:
            import fitz  # type: ignore[import-untyped]

            doc = fitz.open(str(filepath))
        except Exception as e:
            logger.warning(
                "PyMuPDF fitz could not be loaded. Image/figure extraction will be skipped: %s", e
            )

        # pymupdf4llm returns a list of dicts with 'text' and 'metadata' per page
        pages = pymupdf4llm.to_markdown(
            str(filepath),
            page_chunks=True,
        )

        sections: list[DocumentSection] = []

        for page_data in pages:
            page_text: str = (
                page_data.get("text", "") if isinstance(page_data, dict) else str(page_data)
            )
            page_meta: dict = page_data.get("metadata", {}) if isinstance(page_data, dict) else {}
            page_num: int = page_meta.get("page", 0) + 1

            # Extract layout blocks (images/figures) from PyMuPDF
            page_figures: list[FigureRef] = []
            if doc is not None:
                try:
                    page_fitz = doc[page_num - 1]
                    for block in page_fitz.get_text("blocks"):
                        # block format: (x0, y0, x1, y1, "text", block_no, block_type)
                        # block_type 1 represents image/figure blocks
                        if len(block) > 6 and block[6] == 1:
                            page_figures.append(
                                FigureRef(
                                    caption=f"Image {block[5]} on page {page_num}",
                                    page=page_num,
                                    bbox=[
                                        float(block[0]),
                                        float(block[1]),
                                        float(block[2]),
                                        float(block[3]),
                                    ],
                                )
                            )
                except Exception as e:
                    logger.warning(
                        "Failed to extract layout blocks from PDF page %d: %s", page_num, e
                    )

            # Split the page markdown by headings
            page_headings = self._split_page_by_headings(page_text, page_num, filepath.name)

            # Post-process heading sections to extract table structures and associate figures
            for sect in page_headings:
                blocks = _extract_tables_from_text(sect.text)
                for block in blocks:
                    new_sect = DocumentSection(
                        text=block["text"],
                        heading=sect.heading,
                        level=sect.level,
                        page=page_num,
                        parent_headings=sect.parent_headings,
                        content_type=block["type"],
                        table=block.get("table"),
                        figures=page_figures,
                        metadata=sect.metadata.copy(),
                    )
                    sections.append(new_sect)

        if doc is not None:
            doc.close()

        return sections

    def _split_page_by_headings(
        self,
        text: str,
        page_num: int,
        source: str,
    ) -> list[DocumentSection]:
        """Split a single page's markdown text by headings."""
        text = text.strip()
        if not text:
            return []

        headings: list[tuple[int, int, str, int]] = []
        for match in _HEADING_RE.finditer(text):
            level = len(match.group(1))
            title = match.group(2).strip()
            headings.append((match.start(), match.end(), title, level))

        if not headings:
            return [
                DocumentSection(
                    text=text,
                    page=page_num,
                    metadata={"source": source},
                )
            ]

        sections: list[DocumentSection] = []

        # Preamble before first heading
        preamble = text[: headings[0][0]].strip()
        if preamble:
            sections.append(
                DocumentSection(
                    text=preamble,
                    page=page_num,
                    metadata={"source": source},
                )
            )

        heading_stack: list[tuple[int, str]] = []

        for i, (start, end, title, level) in enumerate(headings):
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))

            body_start = end
            body_end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
            body = text[body_start:body_end].strip()

            parent_headings = [h[1] for h in heading_stack[:-1]]

            sections.append(
                DocumentSection(
                    text=body if body else title,
                    heading=title,
                    level=level,
                    page=page_num,
                    parent_headings=parent_headings,
                    metadata={"source": source},
                )
            )

        return sections
