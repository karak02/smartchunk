"""CSV parser — extracts structured sections and tables from CSV files.
"""

from __future__ import annotations

import csv
from pathlib import Path

from smartchunk.models import DocumentSection, TableData
from smartchunk.parsers.base import BaseParser


class CsvParser(BaseParser):
    """Parses Comma-Separated Values (.csv) documents.

    Converts tabular data into a structured TableData representation.
    """

    supported_extensions = [".csv"]

    def parse(self, filepath: str | Path) -> list[DocumentSection]:
        filepath = Path(filepath)

        with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            return []

        headers = [h.strip() for h in rows[0]]
        data_rows = [[cell.strip() for cell in r] for r in rows[1:] if any(cell.strip() for cell in r)]

        table_data = TableData(headers=headers, rows=data_rows, caption=filepath.name)

        return [
            DocumentSection(
                text=table_data.to_markdown(),
                heading=filepath.stem,
                level=1,
                parent_headings=[],
                content_type="table",
                table=table_data,
                metadata={"source": filepath.name},
            )
        ]
