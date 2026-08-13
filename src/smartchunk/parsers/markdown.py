"""Markdown parser — heading-aware section splitting with hierarchy tracking."""

from __future__ import annotations

import re
from pathlib import Path

from smartchunk.models import DocumentSection
from smartchunk.parsers.base import BaseParser

# Matches ATX-style headings: # H1, ## H2, etc.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#*)?$", re.MULTILINE)


class MarkdownParser(BaseParser):
    """Parses Markdown files into heading-delimited sections.

    Preserves the heading hierarchy as ``parent_headings`` so downstream
    chunkers can build a ``parent_context`` string.
    """

    supported_extensions = [".md", ".markdown", ".mdx"]

    def parse(self, filepath: str | Path) -> list[DocumentSection]:
        filepath = Path(filepath)
        text = filepath.read_text(encoding="utf-8")
        return self.parse_text(text, source=filepath.name)

    def parse_text(self, text: str, source: str = "<string>") -> list[DocumentSection]:
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Find all headings and their positions
        headings: list[tuple[int, int, str, int]] = []  # (start, end, title, level)
        for match in _HEADING_RE.finditer(text):
            level = len(match.group(1))
            title = match.group(2).strip()
            headings.append((match.start(), match.end(), title, level))

        if not headings:
            # No headings — treat as plain text
            cleaned = text.strip()
            if not cleaned:
                return []
            return [
                DocumentSection(
                    text=cleaned,
                    metadata={"source": source},
                )
            ]

        sections: list[DocumentSection] = []

        # Text before the first heading (preamble)
        preamble = text[: headings[0][0]].strip()
        if preamble:
            sections.append(
                DocumentSection(
                    text=preamble,
                    metadata={"source": source},
                )
            )

        # Track active heading hierarchy
        heading_stack: list[tuple[int, str]] = []  # (level, title)

        for i, (start, end, title, level) in enumerate(headings):
            # Update heading stack: pop headings at same or deeper level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))

            # Extract body text (everything between this heading and the next)
            body_start = end
            body_end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
            body = text[body_start:body_end].strip()

            # Build parent headings list (everything except current heading)
            parent_headings = [h[1] for h in heading_stack[:-1]]

            sections.append(
                DocumentSection(
                    text=body if body else title,
                    heading=title,
                    level=level,
                    parent_headings=parent_headings,
                    metadata={"source": source},
                )
            )

        return sections
