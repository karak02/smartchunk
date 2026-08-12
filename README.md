# smartchunk

Self-describing document chunks with summaries, entities, and contextual metadata for production RAG workflows.

## Features

- `SmartChunk` class that enriches each chunk with:
  - `summary`
  - `entities`
  - `keywords`
  - `parent_context`
  - `prev_summary`
  - `next_summary`
  - `chunk_type`
  - `confidence_score`
- Chunking backends:
  - `recursive`
  - `semantic` (uses `sentence-transformers` when installed)
  - `fixed-size`
- Export helpers:
  - `to_langchain()`
  - `to_llamaindex()`
  - `to_pinecone()`
  - `to_chroma()`
  - `to_weaviate()`
  - `to_qdrant()`
- API-key-free entity extraction using regex and heuristics
- Extractive summary generation (first sentence / keywords)

## Install

```bash
pip install -e .
```

Optional semantic backend dependencies:

```bash
pip install -e .[semantic]
```

## Quick example

```python
from smartchunk import SmartChunk

text = """
SmartChunk improves context quality in Retrieval-Augmented Generation systems.
It enriches every chunk with summaries, entities, and neighboring context.
"""

chunker = SmartChunk(backend="recursive", chunk_size=180)
chunks = chunker.chunk(text)

print(chunks[0].summary)
print(chunker.to_langchain(chunks)[0])
```

Run the packaged example:

```bash
python /home/runner/work/smartchunk/smartchunk/examples/basic_usage.py
```

## Testing

```bash
python -m unittest discover -s tests
```

## License

MIT
