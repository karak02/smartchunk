"""SmartChunk Multimodal Example — process PDFs extracting text, tables, AND figures.

Demonstrates the full multimodal pipeline:
    Document → Parser → [Text chunks | Table chunks | Figure chunks]
                      → Context Enrichment
                      → Multimodal Embeddings (contextual_text)
                      → Ready for vector DB

Usage:
    pip install smartchunk[pdf]
    python examples/multimodal.py path/to/your.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from smartchunk import SmartChunker


def main(pdf_path: str, enrich: bool = False, model: str = "gpt-4o-mini") -> None:
    print("=" * 60)
    print("SmartChunk Multimodal Pipeline")
    print("=" * 60)
    print(f"Input : {pdf_path}")
    print(f"Enrich: {enrich}\n")

    chunker = SmartChunker(
        model=model,
        chunk_size=512,
        chunk_overlap=50,
        strategy="structural",
        enrich=enrich,
    )

    chunks = chunker.process(pdf_path)

    # ── Content type breakdown ────────────────────────────────────────────────
    type_counts = Counter(c.content_type for c in chunks)
    print("Content breakdown:")
    for ctype, count in type_counts.items():
        icon = {"text": "📝", "table": "📊", "figure": "🖼️", "image": "🖼️"}.get(ctype, "📄")
        print(f"  {icon}  {ctype:<10} {count:>3} chunks")
    print()

    # ── Show contextual embeddings ────────────────────────────────────────────
    print("Contextual Embedding preview (first 3 chunks):")
    print("-" * 60)
    for chunk in chunks[:3]:
        print(f"[{chunk.content_type.upper()}] {chunk.id[:12]}")
        print("  Contextual text sent to embedder:")
        for line in chunk.contextual_text.splitlines():
            print(f"    {line}")
        print()

    # ── Figure / image chunks ─────────────────────────────────────────────────
    figure_chunks = [c for c in chunks if c.content_type in ("figure", "image")]
    if figure_chunks:
        print(f"Figure chunks ({len(figure_chunks)} total):")
        for c in figure_chunks:
            if c.figures:
                fig = c.figures[0]
                print(f"  Page {c.metadata.page} | bbox={fig.bbox} | caption={fig.caption or 'none'}")
    else:
        print("No figures detected in this document.")
    print()

    # ── Table chunks ──────────────────────────────────────────────────────────
    table_chunks = [c for c in chunks if c.content_type == "table"]
    if table_chunks:
        print(f"Table chunks ({len(table_chunks)} total):")
        for c in table_chunks:
            if c.table:
                print(f"  {c.table.rows}r x {c.table.columns}c | headers={c.table.headers[:4]}")
    print()

    # ── Export ─────────────────────────────────────────────────────────────────
    out = Path(pdf_path).stem + "_multimodal_chunks.json"
    SmartChunker.to_json(chunks, out)
    print(f"Exported {len(chunks)} chunks → {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("pdf", help="Path to PDF file")
    p.add_argument("--enrich", action="store_true")
    p.add_argument("--model", default="gpt-4o-mini")
    args = p.parse_args()
    main(args.pdf, args.enrich, args.model)
