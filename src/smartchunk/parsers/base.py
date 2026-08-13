"""Abstract base parser and auto-detection factory."""

from __future__ import annotations

import abc
from pathlib import Path

from smartchunk.models import DocumentSection


class BaseParser(abc.ABC):
    """Base class for all document parsers.

    Subclasses must implement ``parse`` which converts a file path
    into a flat list of :class:`DocumentSection` objects.
    """

    #: File extensions this parser handles (lowercase, with dot).
    supported_extensions: list[str] = []

    @abc.abstractmethod
    def parse(self, filepath: str | Path) -> list[DocumentSection]:
        """Parse a file and return a list of document sections.

        Parameters
        ----------
        filepath:
            Absolute or relative path to the source file.

        Returns
        -------
        list[DocumentSection]
            Ordered list of sections preserving document order.
        """

    def parse_text(self, text: str, source: str = "<string>") -> list[DocumentSection]:
        """Parse raw text directly (not from file).

        Default implementation wraps the text in a single section.
        Subclasses may override for smarter splitting.

        Parameters
        ----------
        text:
            Raw text content.
        source:
            Label for provenance metadata.
        """
        return [
            DocumentSection(
                text=text,
                metadata={"source": source},
            )
        ]


def get_parser(filepath: str | Path) -> BaseParser:
    """Return the appropriate parser for a given file extension.

    Parameters
    ----------
    filepath:
        Path to the file to parse.

    Returns
    -------
    BaseParser
        An instance of the matching parser.

    Raises
    ------
    ValueError
        If no parser is registered for the file extension.
    """
    ext = Path(filepath).suffix.lower()

    if ext in (".txt", ".text", ".log"):
        from smartchunk.parsers.text import TextParser

        return TextParser()

    if ext in (".md", ".markdown", ".mdx"):
        from smartchunk.parsers.markdown import MarkdownParser

        return MarkdownParser()

    if ext in (".pdf",):
        from smartchunk.parsers.pdf import PdfParser

        return PdfParser()

    if ext in (".docx",):
        from smartchunk.parsers.docx import DocxParser

        return DocxParser()

    if ext in (".html", ".htm"):
        from smartchunk.parsers.html_parser import HtmlParser

        return HtmlParser()

    if ext in (".csv",):
        from smartchunk.parsers.csv_parser import CsvParser

        return CsvParser()

    if ext in (".json",):
        from smartchunk.parsers.json_parser import JsonParser

        return JsonParser()

    if ext in (".xml",):
        from smartchunk.parsers.xml_parser import XmlParser

        return XmlParser()

    if ext in (".epub",):
        from smartchunk.parsers.epub import EpubParser

        return EpubParser()

    if ext in (".pptx",):
        from smartchunk.parsers.pptx import PptxParser

        return PptxParser()

    if ext in (".xlsx",):
        from smartchunk.parsers.xlsx import XlsxParser

        return XlsxParser()

    raise ValueError(
        f"No parser registered for extension '{ext}'. "
        "Supported: .txt, .text, .log, .md, .markdown, .mdx, .pdf, .docx, "
        ".html, .htm, .csv, .json, .xml, .epub, .pptx, .xlsx"
    )
