"""Tests for TableData formatting and table preservation in chunkers."""

from smartchunk.chunkers.recursive import RecursiveChunker
from smartchunk.models import DocumentSection, TableData


def test_table_data_formatting():
    table = TableData(
        headers=["Product", "Revenue", "Growth"],
        rows=[["A", "$10M", "12%"], ["B", "$7M", "18%"]],
        caption="Q3 Summary",
    )

    md = table.to_markdown()
    assert "| Product | Revenue | Growth |" in md
    assert "| A | $10M | 12% |" in md

    text = table.to_text()
    assert "Product: A, Revenue: $10M, Growth: 12%" in text


def test_chunker_preserves_table():
    table = TableData(
        headers=["Item", "Cost"],
        rows=[["Server", "$5,000"], ["License", "$1,000"]],
    )

    section = DocumentSection(
        text="Table text",
        content_type="table",
        table=table,
    )

    chunker = RecursiveChunker()
    chunks = chunker.chunk([section])

    assert len(chunks) == 1
    assert chunks[0]["content_type"] == "table"
    assert chunks[0]["table"] == table
