"""Tests for DocxParser and parser factory detection."""

from smartchunk.parsers.base import get_parser
from smartchunk.parsers.docx import DocxParser


def test_docx_parser_factory_detection():
    parser = get_parser("document.docx")
    assert isinstance(parser, DocxParser)
    assert ".docx" in parser.supported_extensions
