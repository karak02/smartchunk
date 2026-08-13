"""SmartChunk Basic Example — process text and a plain text file.

Usage:
    python examples/basic.py
    python examples/basic.py --enrich --model gpt-4o-mini
    python examples/basic.py --enrich --model ollama/qwen2.5-coder:1.5b
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from smartchunk import SmartChunker

# ── 1. Zero-config: no API key needed ────────────────────────────────────────
print("=" * 60)
print("EXAMPLE 1 — No enrichment (zero config, no API key)")
print("=" * 60)

chunker = SmartChunker(enrich=False)
chunks = chunker.process_text(
    """
    SmartChunk is a Python library that turns raw document text into
    self-describing chunks for production RAG pipelines.

    Every chunk carries its own summary, entities, keywords, and parent
    context so your retrieval engine has 10x more signal without extra
    embedding calls.

    It supports PDF, DOCX, HTML, CSV, XLSX, PPTX, Markdown, EPUB,
    JSON, XML, TXT, and LOG files out of the box.
    """,
    source="intro.txt",
)
for chunk in chunks:
    print(f"  [{chunk.id[:8]}] {chunk.text[:80].strip()}...")
    print(f"           tokens={chunk.metadata.token_count}, context={chunk.parent_context or 'root'}")
print()

# ── 2. With enrichment ────────────────────────────────────────────────────────
print("=" * 60)
print("EXAMPLE 2 — With LLM enrichment (set OPENAI_API_KEY or use Ollama)")
print("=" * 60)

# Uncomment and set your model:
# chunker = SmartChunker(model="gpt-4o-mini", enrich=True)
# chunker = SmartChunker(model="ollama/llama3", enrich=True)
# chunks = chunker.process("examples/sample.txt")
# for chunk in chunks:
#     print(f"  Summary:    {chunk.summary}")
#     print(f"  Entities:   {chunk.entities}")
#     print(f"  Keywords:   {chunk.keywords}")
#     print(f"  Confidence: {chunk.confidence:.2f}")
#     print()
print("  (Uncomment block above after setting your API key or Ollama)")
