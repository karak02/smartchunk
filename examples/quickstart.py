"""SmartChunk Quickstart — process a document and explore the results.

Usage:
    python examples/quickstart.py

    # With enrichment (requires LLM API key):
    OPENAI_API_KEY=sk-... python examples/quickstart.py --enrich

    # Export to JSON:
    python examples/quickstart.py --output output.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smartchunk import SmartChunker


def main():
    parser = argparse.ArgumentParser(description="SmartChunk quickstart demo")
    parser.add_argument(
        "file",
        nargs="?",
        default=str(Path(__file__).parent / "sample.txt"),
        help="Path to document to process (default: examples/sample.txt)",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Enable LLM enrichment (requires API key)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model for enrichment (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--strategy",
        choices=["recursive", "semantic", "structural"],
        default="recursive",
        help="Chunking strategy (default: recursive)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Target chunk size in tokens (default: 256)",
    )
    parser.add_argument(
        "--output",
        help="Export chunks to JSON file",
    )
    args = parser.parse_args()

    # ── Create chunker ─────────────────────────────────────────
    print(f"[CONFIG] Strategy: {args.strategy}")
    print(f"[CONFIG] Chunk size: {args.chunk_size} tokens")
    print(f"[CONFIG] Enrichment: {'ON' if args.enrich else 'OFF'}")
    print()

    chunker = SmartChunker(
        model=args.model,
        chunk_size=args.chunk_size,
        strategy=args.strategy,
        enrich=args.enrich,
    )

    # ── Process ────────────────────────────────────────────────
    filepath = Path(args.file)
    print(f"[PROCESS] Processing: {filepath.name}")
    chunks = chunker.process(filepath)
    print(f"[DONE] Created {len(chunks)} chunks\n")

    # ── Display results ────────────────────────────────────────
    for i, chunk in enumerate(chunks):
        print(f"{'=' * 60}")
        print(f"  Chunk {i + 1}/{len(chunks)}")
        print(f"   ID:       {chunk.id}")
        print(f"   Tokens:   {chunk.metadata.token_count}")
        print(f"   Context:  {chunk.parent_context or '(none)'}")
        print(f"   Text:     {chunk.text[:120]}{'...' if len(chunk.text) > 120 else ''}")

        if args.enrich:
            print(f"   Summary:  {chunk.summary}")
            print(f"   Entities: {chunk.entities}")
            print(f"   Keywords: {chunk.keywords}")
            print(f"   Confidence: {chunk.confidence:.2f}")
            print(f"   Prev:     {chunk.prev_summary or '(start)'}")
            print(f"   Next:     {chunk.next_summary or '(end)'}")

        print()

    # ── Export ─────────────────────────────────────────────────
    if args.output:
        SmartChunker.to_json(chunks, args.output)
        print(f"[EXPORT] Exported to {args.output}")

    # ── Usage stats ────────────────────────────────────────────
    if args.enrich:
        stats = chunker.usage_stats
        print(f"\n[USAGE] {stats['total_tokens']} tokens, ~${stats['estimated_cost_usd']:.4f}")


if __name__ == "__main__":
    main()
