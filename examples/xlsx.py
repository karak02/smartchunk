"""SmartChunk XLSX Example — parse spreadsheets preserving table structure.

Usage:
    pip install smartchunk[xlsx]
    python examples/xlsx.py path/to/your.xlsx
    python examples/xlsx.py path/to/your.xlsx --enrich --model gpt-4o-mini
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from smartchunk import SmartChunker


def main(xlsx_path: str, enrich: bool = False, model: str = "gpt-4o-mini") -> None:
    print(f"Processing XLSX: {xlsx_path}")

    # SmartChunk preserves every sheet as a structured TableData object.
    # Rows and headers are kept intact — no data is lost to text truncation.
    chunker = SmartChunker(
        model=model,
        enrich=enrich,
        strategy="recursive",
    )

    chunks = chunker.process(xlsx_path)
    table_chunks = [c for c in chunks if c.content_type == "table"]

    print(f"\nFound {len(table_chunks)} table chunks across {xlsx_path}\n")

    for i, chunk in enumerate(table_chunks, 1):
        print(f"=== Table Chunk {i} ===")
        print(f"  Sheet / source : {chunk.metadata.source}")
        print(f"  Chunk ID       : {chunk.id}")
        if chunk.table:
            print(f"  Dimensions     : {chunk.table.rows} rows x {chunk.table.columns} cols")
            print(f"  Headers        : {chunk.table.headers}")
            if chunk.table.data:
                print("  First row      :", chunk.table.data[0])
        if enrich:
            print(f"  Summary        : {chunk.summary}")
            print(f"  Entities       : {chunk.entities}")
            print(f"  Keywords       : {chunk.keywords}")
        print()

    # Export
    out = Path(xlsx_path).stem + "_chunks.json"
    SmartChunker.to_json(chunks, out)
    print(f"All chunks exported to: {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("xlsx", help="Path to XLSX file")
    p.add_argument("--enrich", action="store_true")
    p.add_argument("--model", default="gpt-4o-mini")
    args = p.parse_args()
    main(args.xlsx, args.enrich, args.model)
