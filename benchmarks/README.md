# SmartChunk Benchmarks

This directory contains benchmark scripts comparing SmartChunk vs raw chunking
on standard RAG evaluation benchmarks.

## Planned Benchmarks

- **Retrieval accuracy** (Hit@k) on NarrativeQA, QuALITY, and QASPER
- **Chunk quality** (coherence, atomicity, information density)
- **Token efficiency** (tokens per retrieved answer)
- **End-to-end latency** (parse → chunk → enrich → embed → retrieve)

## Run

```bash
pip install smartchunk[all]
python benchmarks/run.py --dataset narrativeqa --model gpt-4o-mini
```
