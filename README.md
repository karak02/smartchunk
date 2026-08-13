<p align="center">
  <img src="https://raw.githubusercontent.com/YOUR_USERNAME/smartchunk/main/docs/assets/banner.png" alt="SmartChunk" width="100%">
</p>

<h1 align="center">⚡ SmartChunk</h1>
<p align="center"><strong>Self-describing document chunks for production RAG</strong></p>

<p align="center">
  <a href="https://your-demo-url.com"><img src="https://img.shields.io/badge/🚀 Live Demo-Try It-22c55e?style=for-the-badge" alt="Demo"></a>
  &nbsp;
  <a href="https://pypi.org/project/smartchunk/"><img src="https://img.shields.io/pypi/v/smartchunk?style=for-the-badge&color=0d6efd" alt="PyPI"></a>
  &nbsp;
  <a href="https://github.com/YOUR_USERNAME/smartchunk/actions"><img src="https://img.shields.io/github/actions/workflow/status/YOUR_USERNAME/smartchunk/ci.yml?style=for-the-badge&label=CI" alt="CI"></a>
  &nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="MIT License"></a>
  &nbsp;
  <a href="https://pypi.org/project/smartchunk/"><img src="https://img.shields.io/pypi/pyversions/smartchunk?style=for-the-badge" alt="Python"></a>
</p>

<p align="center">
  <a href="#-the-problem">Problem</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-installation">Install</a> •
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-demo">Demo</a> •
  <a href="#-benchmarks">Benchmarks</a> •
  <a href="docs/">Docs</a>
</p>

---

## 🎬 Demo

https://github.com/YOUR_USERNAME/smartchunk/assets/YOUR_ASSET_ID/your-demo-video.mp4

> Upload any document → See chunks enriched with summaries, entities, keywords, and contextual embeddings in real time.

---

## ❌ The Problem

You split a document into chunks and push them to a vector database.

When a user asks *"What did the board decide about the Q3 budget?"* — your retriever pulls chunks containing "board" and "budget" but:

- **Misses context** — the relevant sentence is split across two chunks
- **Loses hierarchy** — the chunk has no idea it came from the "Financial Strategy" section
- **Ignores neighbors** — the preceding chunk has the subject, the next has the conclusion
- **Keyword queries fail** — "Q3 budget" doesn't match "third-quarter capital allocation"

**The result: poor recall, hallucinations, and frustrated users.**

---

## ✅ The Solution

SmartChunk turns each raw chunk into a **self-describing knowledge unit**:

```python
{
    "text": "The board approved a $50M expansion...",

    # ── LLM Enrichment ────────────────────────────────────────
    "summary":        "Board approves major capital expenditure",
    "entities":       ["Board of Directors", "$50M", "Q3 2026"],
    "keywords":       ["budget", "expansion", "capital expenditure", "CAPEX"],
    "confidence":     0.94,   # how self-contained is this chunk?

    # ── Structural Context ────────────────────────────────────
    "parent_context": "Annual Report → Financial Strategy → Capital Allocation",
    "prev_summary":   "CEO outlines growth plan for APAC region",
    "next_summary":   "Risk factors and mitigation strategies discussed",

    # ── Contextual Embedding Text ─────────────────────────────
    "contextual_text": """Annual Report 2026
Financial Strategy → Capital Allocation
Board approves major capital expenditure
The board approved a $50M expansion...""",

    # ── Metadata ──────────────────────────────────────────────
    "metadata": {
        "source": "annual_report.pdf",
        "page": 12,
        "chunk_index": 7,
        "total_chunks": 42,
        "token_count": 128
    }
}
```

---

## 🏗 Architecture

```
📄 Document
    │
    ▼
┌─────────────────────────────────┐
│           Parser                │
│  PDF · DOCX · HTML · MD · PPTX  │
│  CSV · XLSX · JSON · XML · EPUB │
└──────────────┬──────────────────┘
               │
    ┌──────────▼──────────┐
    │  Text  │  Table  │  │
    │ chunks │ chunks  │  │
    │        │         │  │
    │  Figure/Image chunks│
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │   Context Enrichment│
    │  (single LLM call)  │
    │  summary · entities │
    │  keywords · confid. │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ Contextual Embedding│
    │ section + summary + │
    │ raw text prepended  │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │   Hybrid Retrieval  │
    │ Dense + BM25 + RRF  │
    │ + Metadata Filters  │
    │ + Reranking         │
    └─────────────────────┘
```

---

## 📦 Installation

```bash
# Core (required)
pip install smartchunk

# With PDF support
pip install smartchunk[pdf]

# With semantic chunking
pip install smartchunk[semantic]

# With XLSX support
pip install smartchunk[xlsx]

# With vector DB export
pip install smartchunk[pinecone]
pip install smartchunk[chromadb]

# Everything
pip install smartchunk[all]

# Development
pip install smartchunk[all,dev]
```

---

## ⚡ Quickstart

### No API key needed

```python
from smartchunk import SmartChunker

chunker = SmartChunker(enrich=False)
chunks = chunker.process("document.pdf")

for chunk in chunks:
    print(chunk.text)
    print(chunk.parent_context)   # "Introduction → Background"
    print(chunk.metadata.page)    # 3
```

### With LLM enrichment

```python
chunker = SmartChunker(
    model="gpt-4o-mini",     # or "ollama/llama3" for local
    chunk_size=512,
    strategy="structural",   # best for PDFs with headings
    enrich=True,
)

chunks = chunker.process("annual_report.pdf")

for chunk in chunks:
    print(chunk.summary)          # "Board approves $50M expansion"
    print(chunk.entities)         # ["Board of Directors", "$50M"]
    print(chunk.keywords)         # ["capital", "expansion", "CAPEX"]
    print(chunk.confidence)       # 0.94
    print(chunk.contextual_text)  # Ready-to-embed enriched text
```

### Local LLM (Ollama — free, no API key)

```bash
ollama pull qwen2.5-coder:1.5b
```
```python
chunker = SmartChunker(model="ollama/qwen2.5-coder:1.5b", enrich=True)
chunks = chunker.process("document.pdf")
```

### Export to vector DB

```python
# Pinecone
SmartChunker.to_pinecone(chunks, index_name="my-index")

# ChromaDB
SmartChunker.to_chromadb(chunks, collection="my-docs")

# JSON / JSONL
SmartChunker.to_json(chunks, "output.json")
SmartChunker.to_jsonl(chunks, "output.jsonl")
```

---

## 🔧 Chunking Strategies

| Strategy | Best For | How It Works |
|:---|:---|:---|
| `recursive` *(default)* | General text | Splits on `\n\n` → `\n` → `. ` → ` `, respects token limits |
| `semantic` | Topic-dense docs | Embeds sentences, splits where cosine similarity drops |
| `structural` | Headed documents | Uses headings as boundaries, recursive fallback for large sections |

---

## 🧠 Supported File Formats

| Format | Extension | Notes |
|:---|:---|:---|
| PDF | `.pdf` | Text, tables, figures, images — requires `smartchunk[pdf]` |
| Word | `.docx` | Paragraphs, tables, headings |
| PowerPoint | `.pptx` | Slides as structured sections |
| Excel | `.xlsx` | Sheets as `TableData` objects |
| HTML | `.html`, `.htm` | Tag stripping, link preservation |
| Markdown | `.md`, `.mdx` | Heading hierarchy preserved |
| CSV | `.csv` | Tabular data as `TableData` |
| EPUB | `.epub` | Chapter-aware extraction |
| JSON | `.json` | Key-value flattening |
| XML | `.xml` | Tag-aware normalization |
| Plain text | `.txt`, `.log` | Paragraph-based splitting |

---

## 🔍 Advanced Retrieval

SmartChunk includes a built-in hybrid retrieval engine:

```python
from smartchunk.retrieval import HybridRetriever

retriever = HybridRetriever(chunks)

results = retriever.search(
    query="What was the Q3 capital allocation?",
    top_k=5,
    use_bm25=True,          # lexical matching for exact terms
    metadata_filter={"source": "annual_report.pdf"},
)

for result in results:
    print(result.chunk.text)
    print(result.score)
```

**How it works:**
- **Dense retrieval** — cosine similarity on `contextual_text` embeddings
- **BM25** — keyword matching for names, IDs, dates, amounts
- **RRF fusion** — Reciprocal Rank Fusion combines both signals
- **Reranking** — optional cross-encoder reranking stage

---

## 📊 vs. Raw Chunking

| Feature | Raw Chunking | SmartChunk |
|:---|:---:|:---:|
| Text splitting | ✅ | ✅ |
| Token-aware sizing | ⚠️ | ✅ |
| Section hierarchy | ❌ | ✅ `parent_context` |
| Per-chunk summary | ❌ | ✅ |
| Named entity extraction | ❌ | ✅ |
| Keyword enrichment | ❌ | ✅ |
| Neighbor linking | ❌ | ✅ `prev/next_summary` |
| Atomicity scoring | ❌ | ✅ `confidence` |
| Contextual embeddings | ❌ | ✅ `contextual_text` |
| Hybrid retrieval | ❌ | ✅ Dense + BM25 + RRF |
| Reranking stage | ❌ | ✅ |
| Table preservation | ❌ | ✅ `TableData` |
| Multimodal (figures) | ❌ | ✅ `FigureRef` |
| One-line DB export | ❌ | ✅ Pinecone, ChromaDB |
| LLM cache | ❌ | ✅ SHA-256 hash cache |

---

## 🖥 Developer Dashboard

Run the interactive demo locally — no API key required (works with Ollama):

```bash
python run_demo.py
```

Opens at `http://localhost:8000` — upload any document and see the full pipeline live:

- **Compare strategies** side-by-side (recursive, semantic, structural)
- **Inspect chunks** — text, summary, entities, keywords, parent context, contextual embedding
- **Monitor cache** — hit rate, time saved, LLM calls avoided
- **Multimodal view** — text, table, and figure chunks displayed separately

---

## ⚙️ Configuration

```bash
# .env
SMARTCHUNK_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...

SMARTCHUNK_CHUNK_SIZE=512
SMARTCHUNK_CHUNK_OVERLAP=50
SMARTCHUNK_STRATEGY=recursive

SMARTCHUNK_ENRICH=true
SMARTCHUNK_BATCH_SIZE=10
SMARTCHUNK_MAX_CONCURRENCY=5
```

Supported models — anything from [LiteLLM](https://docs.litellm.ai/docs/providers):

```python
SmartChunker(model="gpt-4o-mini")              # OpenAI
SmartChunker(model="claude-3-haiku-20240307")  # Anthropic
SmartChunker(model="ollama/llama3")            # Local (free)
SmartChunker(model="gemini/gemini-1.5-flash")  # Google
```

---

## 📚 API Reference

### `SmartChunker`

| Method | Description |
|:---|:---|
| `process(filepath)` | Parse + chunk + enrich a file |
| `process_text(text, source)` | Process raw text |
| `to_json(chunks, path)` | Export to JSON |
| `to_jsonl(chunks, path)` | Export to JSONL |
| `to_dict(chunks)` | Convert to list of dicts |
| `to_pinecone(chunks, index_name)` | Export to Pinecone |
| `to_chromadb(chunks, collection)` | Export to ChromaDB |
| `usage_stats` | Token usage + cost |

### `SmartChunk` Fields

| Field | Type | Description |
|:---|:---|:---|
| `id` | `str` | Unique chunk ID |
| `text` | `str` | Raw chunk text |
| `contextual_text` | `str` | Enriched text for embedding |
| `summary` | `str` | One-sentence summary |
| `entities` | `list[str]` | People, orgs, amounts, dates |
| `keywords` | `list[str]` | Semantic retrieval keywords |
| `parent_context` | `str` | Section hierarchy path |
| `prev_summary` | `str` | Previous chunk summary |
| `next_summary` | `str` | Next chunk summary |
| `confidence` | `float` | Atomicity score (0–1) |
| `content_type` | `str` | `text`, `table`, `figure`, `image` |
| `table` | `TableData \| None` | Structured table (rows, headers) |
| `figures` | `list[FigureRef]` | Detected figures/images |
| `metadata` | `ChunkMetadata` | Source, page, tokens |

---

## 🤝 Contributing

```bash
git clone https://github.com/YOUR_USERNAME/smartchunk.git
cd smartchunk
pip install -e ".[all,dev]"
pytest tests/ -v
```

PRs welcome. Please open an issue first for large changes.

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
  Built for developers who are serious about RAG quality.
  <br>
  <strong>Star ⭐ the repo if SmartChunk helps your pipeline.</strong>
</p>
