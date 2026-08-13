"""Tests for document parsers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from smartchunk.models import DocumentSection
from smartchunk.parsers.base import get_parser
from smartchunk.parsers.markdown import MarkdownParser
from smartchunk.parsers.text import TextParser


class TestTextParser:
    """Tests for the plain text parser."""

    def test_parse_splits_on_blank_lines(self, sample_text: str):
        parser = TextParser()
        sections = parser.parse_text(sample_text)

        assert len(sections) == 4
        assert all(isinstance(s, DocumentSection) for s in sections)

    def test_parse_normalises_whitespace(self):
        parser = TextParser()
        sections = parser.parse_text("  hello   world  \n\n  foo   bar  ")

        assert len(sections) == 2
        assert sections[0].text == "hello world"
        assert sections[1].text == "foo bar"

    def test_parse_empty_text(self):
        parser = TextParser()
        sections = parser.parse_text("")
        assert sections == []

    def test_parse_from_file(self, tmp_path: Path):
        file = tmp_path / "test.txt"
        file.write_text("Paragraph one.\n\nParagraph two.", encoding="utf-8")

        parser = TextParser()
        sections = parser.parse(file)

        assert len(sections) == 2
        assert sections[0].text == "Paragraph one."
        assert sections[1].text == "Paragraph two."

    def test_parse_preserves_source(self, tmp_path: Path):
        file = tmp_path / "myfile.txt"
        file.write_text("Hello world.", encoding="utf-8")

        parser = TextParser()
        sections = parser.parse(file)

        assert sections[0].metadata["source"] == "myfile.txt"


class TestMarkdownParser:
    """Tests for the markdown parser."""

    def test_parse_heading_hierarchy(self, sample_markdown: str):
        parser = MarkdownParser()
        sections = parser.parse_text(sample_markdown)

        # Should have: preamble (exec summary) + headings
        assert len(sections) >= 5

        # Find the "Capital Allocation" section
        cap_alloc = [s for s in sections if s.heading == "Capital Allocation"]
        assert len(cap_alloc) == 1
        assert cap_alloc[0].level == 3
        assert "Financial Strategy" in cap_alloc[0].parent_headings

    def test_parse_preserves_heading_text(self, sample_markdown: str):
        parser = MarkdownParser()
        sections = parser.parse_text(sample_markdown)

        headings = [s.heading for s in sections if s.heading]
        assert "Annual Report 2026" in headings
        assert "Financial Strategy" in headings
        assert "Risk Factors" in headings

    def test_parse_no_headings(self):
        parser = MarkdownParser()
        sections = parser.parse_text("Just plain text without any headings.")

        assert len(sections) == 1
        assert sections[0].heading is None

    def test_parse_empty_markdown(self):
        parser = MarkdownParser()
        sections = parser.parse_text("")
        assert sections == []

    def test_parse_from_file(self, tmp_path: Path, sample_markdown: str):
        file = tmp_path / "test.md"
        file.write_text(sample_markdown, encoding="utf-8")

        parser = MarkdownParser()
        sections = parser.parse(file)

        assert len(sections) >= 5


class TestGetParser:
    """Tests for the parser auto-detection factory."""

    def test_txt_detection(self, tmp_path: Path):
        parser = get_parser(tmp_path / "test.txt")
        assert isinstance(parser, TextParser)

    def test_md_detection(self, tmp_path: Path):
        parser = get_parser(tmp_path / "README.md")
        assert isinstance(parser, MarkdownParser)

    def test_log_detection(self, tmp_path: Path):
        parser = get_parser(tmp_path / "app.log")
        assert isinstance(parser, TextParser)

    def test_unsupported_extension(self, tmp_path: Path):
        with pytest.raises(ValueError, match="No parser registered"):
            get_parser(tmp_path / "data.xyz")

    def test_pdf_detection(self, tmp_path: Path):
        # PDF parser import — just test factory, not actual parsing
        from smartchunk.parsers.pdf import PdfParser

        parser = get_parser(tmp_path / "report.pdf")
        assert isinstance(parser, PdfParser)

    def test_csv_parser(self, tmp_path: Path):
        from smartchunk.parsers.csv_parser import CsvParser
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("Name,Age,Role\nAlice,30,Engineer\nBob,25,Designer\n", encoding="utf-8")
        
        parser = CsvParser()
        sections = parser.parse(csv_file)
        assert len(sections) == 1
        assert sections[0].content_type == "table"
        assert sections[0].table is not None
        assert sections[0].table.headers == ["Name", "Age", "Role"]
        assert sections[0].table.rows == [["Alice", "30", "Engineer"], ["Bob", "25", "Designer"]]

    def test_json_parser(self, tmp_path: Path):
        from smartchunk.parsers.json_parser import JsonParser
        json_file = tmp_path / "test.json"
        json_file.write_text('{"project": "SmartChunk", "version": 1.0}', encoding="utf-8")
        
        parser = JsonParser()
        sections = parser.parse(json_file)
        assert len(sections) == 1
        assert "SmartChunk" in sections[0].text
        assert "version" in sections[0].text

    def test_xml_parser(self, tmp_path: Path):
        from smartchunk.parsers.xml_parser import XmlParser
        xml_file = tmp_path / "test.xml"
        xml_file.write_text("<root><child>Hello World</child></root>", encoding="utf-8")
        
        parser = XmlParser()
        sections = parser.parse(xml_file)
        assert len(sections) == 1
        assert sections[0].text.strip() == "Hello World"

    def test_pdf_multimodal_extraction(self, tmp_path: Path):
        from unittest.mock import MagicMock, patch
        from smartchunk.parsers.pdf import PdfParser

        pdf_file = tmp_path / "dummy.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy content")

        mock_pages = [
            {
                "text": "Some preamble text.\n\n| Column A | Column B |\n| --- | --- |\n| Value 1 | Value 2 |\n\nSome postamble text.",
                "metadata": {"page": 0}
            }
        ]

        mock_doc = MagicMock()
        mock_page = MagicMock()
        # mock block: (x0, y0, x1, y1, text, block_no, block_type)
        # block_type=1 is image block
        mock_page.get_text.return_value = [
            (50.0, 100.0, 200.0, 250.0, "", 0, 1)
        ]
        mock_doc.__getitem__.return_value = mock_page

        with patch("pymupdf4llm.to_markdown", return_value=mock_pages), \
             patch("fitz.open", return_value=mock_doc):
            parser = PdfParser()
            sections = parser.parse(pdf_file)

            # Expect 3 sections: preamble text, table, postamble text
            assert len(sections) == 3

            assert sections[0].content_type == "text"
            assert "preamble" in sections[0].text
            assert sections[2].content_type == "text"
            assert "postamble" in sections[2].text

            assert sections[1].content_type == "table"
            assert sections[1].table is not None
            assert sections[1].table.headers == ["Column A", "Column B"]
            assert sections[1].table.rows == [["Value 1", "Value 2"]]

            # Check figure extraction
            assert len(sections[0].figures) == 1
            fig = sections[0].figures[0]
            assert fig.page == 1
            assert list(fig.bbox) == [50.0, 100.0, 200.0, 250.0]

