"""SmartChunk PDF Example — extract text, tables, and figures from a PDF.

Usage:
    pip install smartchunk[pdf]
    python examples/pdf.py path/to/your.pdf
    python examples/pdf.py path/to/your.pdf --enrich --model gpt-4o-mini
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from smartchunk import SmartChunker


def main(pdf_path: str, enrich: bool = False, model: str = "gpt-4o-mini") -> None:
    print(f"Processing: {pdf_path}")

    chunker = SmartChunker(
        model=model,
        chunk_size=512,
        chunk_overlap=50,
        strategy="structural",  # structural is best for PDFs with headings
        enrich=enrich,
    )

    chunks = chunker.process(pdf_path)

    text_chunks = [c for c in chunks if c.content_type == "text"]
    table_chunks = [c for c in chunks if c.content_type == "table"]
    image_chunks = [c for c in chunks if c.content_type in ("figure", "image")]

    print(f"\nExtracted {len(chunks)} total chunks:")
    print(f"  • {len(text_chunks)}  text chunks")
    print(f"  • {len(table_chunks)} table chunks")
    print(f"  • {len(image_chunks)} image/figure chunks\n")

    for i, chunk in enumerate(chunks[:5], 1):
        print(f"--- Chunk {i} [{chunk.content_type.upper()}] ---")
        print(f"  ID:      {chunk.id}")
        print(f"  Source:  {chunk.metadata.source}  page={chunk.metadata.page}")
        print(f"  Tokens:  {chunk.metadata.token_count}")
        print(f"  Context: {chunk.parent_context or 'root'}")
        if chunk.content_type == "table" and chunk.table:
            print(f"  Table:   {chunk.table.rows} rows x {chunk.table.columns} cols")
            print(f"  Headers: {chunk.table.headers}")
        else:
            print(f"  Text:    {chunk.text[:100].strip()}...")
        if enrich:
            print(f"  Summary: {chunk.summary}")
            print(f"  Entities:{chunk.entities}")
        print()

    # Export full results to JSON
    out = Path(pdf_path).stem + "_chunks.json"
    SmartChunker.to_json(chunks, out)
    print(f"Exported all chunks to: {out}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("pdf", help="Path to PDF file")
    p.add_argument("--enrich", action="store_true")
    p.add_argument("--model", default="gpt-4o-mini")
    args = p.parse_args()
    main(args.pdf, args.enrich, args.model)
