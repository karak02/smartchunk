"""Core smart chunking implementation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import re
from typing import Callable, Iterable, List, Optional

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


@dataclass
class EnrichedChunk:
    """Representation of an enriched chunk."""

    text: str
    summary: str
    entities: List[str]
    keywords: List[str]
    parent_context: str
    prev_summary: str
    next_summary: str
    chunk_type: str
    confidence_score: float


class SmartChunk:
    """Split text and enrich each chunk for RAG."""

    def __init__(
        self,
        backend: str = "recursive",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        semantic_similarity_threshold: float = 0.72,
        semantic_max_sentences: int = 4,
    ) -> None:
        self.backend = backend
        self.chunk_size = max(50, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))
        self.semantic_similarity_threshold = semantic_similarity_threshold
        self.semantic_max_sentences = max(1, semantic_max_sentences)
        self._last_chunks: List[EnrichedChunk] = []

    def chunk(self, text: str) -> List[EnrichedChunk]:
        """Chunk and enrich text."""
        text = _normalize_whitespace(text)
        if not text:
            self._last_chunks = []
            return []

        raw_chunks = self._split_text(text)
        parent_context = self._build_parent_context(text)

        enriched = [
            EnrichedChunk(
                text=chunk,
                summary=self._generate_summary(chunk),
                entities=self._extract_entities(chunk),
                keywords=self._extract_keywords(chunk),
                parent_context=parent_context,
                prev_summary="",
                next_summary="",
                chunk_type=self.backend,
                confidence_score=self._confidence_score(chunk),
            )
            for chunk in raw_chunks
            if chunk.strip()
        ]

        for i, item in enumerate(enriched):
            item.prev_summary = enriched[i - 1].summary if i > 0 else ""
            item.next_summary = enriched[i + 1].summary if i < len(enriched) - 1 else ""

        self._last_chunks = enriched
        return enriched

    def __call__(self, text: str) -> List[EnrichedChunk]:
        return self.chunk(text)

    def to_langchain(self, chunks: Optional[Iterable[EnrichedChunk]] = None) -> List[dict]:
        """Export chunks to LangChain document-like objects."""
        return [
            {"page_content": item.text, "metadata": _chunk_metadata(item)}
            for item in self._resolve_chunks(chunks)
        ]

    def to_llamaindex(self, chunks: Optional[Iterable[EnrichedChunk]] = None) -> List[dict]:
        """Export chunks to LlamaIndex node-like objects."""
        return [
            {"text": item.text, "metadata": _chunk_metadata(item)}
            for item in self._resolve_chunks(chunks)
        ]

    def to_pinecone(self, chunks: Optional[Iterable[EnrichedChunk]] = None) -> List[dict]:
        """Export chunks to Pinecone upsert format."""
        result = []
        for idx, item in enumerate(self._resolve_chunks(chunks)):
            result.append(
                {
                    "id": f"chunk-{idx}",
                    "values": [],
                    "metadata": {"text": item.text, **_chunk_metadata(item)},
                }
            )
        return result

    def to_chroma(self, chunks: Optional[Iterable[EnrichedChunk]] = None) -> dict:
        """Export chunks to Chroma add() format."""
        ids, docs, metas = [], [], []
        for idx, item in enumerate(self._resolve_chunks(chunks)):
            ids.append(f"chunk-{idx}")
            docs.append(item.text)
            metas.append(_chunk_metadata(item))
        return {"ids": ids, "documents": docs, "metadatas": metas}

    def to_weaviate(self, chunks: Optional[Iterable[EnrichedChunk]] = None) -> List[dict]:
        """Export chunks to Weaviate object format."""
        return [
            {
                "class": "SmartChunk",
                "properties": {"text": item.text, **_chunk_metadata(item)},
            }
            for item in self._resolve_chunks(chunks)
        ]

    def to_qdrant(self, chunks: Optional[Iterable[EnrichedChunk]] = None) -> List[dict]:
        """Export chunks to Qdrant point format."""
        return [
            {"id": idx, "vector": [], "payload": {"text": item.text, **_chunk_metadata(item)}}
            for idx, item in enumerate(self._resolve_chunks(chunks))
        ]

    def _resolve_chunks(self, chunks: Optional[Iterable[EnrichedChunk]]) -> List[EnrichedChunk]:
        if chunks is None:
            return list(self._last_chunks)
        return list(chunks)

    def _split_text(self, text: str) -> List[str]:
        strategies: dict[str, Callable[[str], List[str]]] = {
            "recursive": self._recursive_split,
            "fixed-size": self._fixed_size_split,
            "fixed_size": self._fixed_size_split,
            "semantic": self._semantic_split,
        }
        try:
            return strategies[self.backend](text)
        except KeyError as exc:
            raise ValueError(
                "Unsupported backend. Use one of: recursive, semantic, fixed-size"
            ) from exc

    def _recursive_split(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        chunks: List[str] = []

        for paragraph in paragraphs or [text]:
            if len(paragraph) <= self.chunk_size:
                chunks.append(paragraph)
                continue

            sentences = _split_sentences(paragraph)
            current = ""
            for sentence in sentences:
                candidate = f"{current} {sentence}".strip()
                if current and len(candidate) > self.chunk_size:
                    chunks.append(current)
                    current = sentence
                else:
                    current = candidate
            if current:
                chunks.append(current)

        return _merge_short_chunks(chunks, self.chunk_size // 3)

    def _fixed_size_split(self, text: str) -> List[str]:
        chunks: List[str] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for start in range(0, len(text), step):
            part = text[start : start + self.chunk_size].strip()
            if part:
                chunks.append(part)
        return chunks

    def _semantic_split(self, text: str) -> List[str]:
        sentences = _split_sentences(text)
        if not sentences:
            return [text]

        embeddings = self._sentence_embeddings(sentences)
        if embeddings is None:
            return self._recursive_split(text)

        chunks: List[str] = []
        current = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
            if (
                sim < self.semantic_similarity_threshold
                or len(current) >= self.semantic_max_sentences
            ):
                chunks.append(" ".join(current).strip())
                current = [sentences[i]]
            else:
                current.append(sentences[i])

        if current:
            chunks.append(" ".join(current).strip())

        return _merge_short_chunks(chunks, self.chunk_size // 4)

    def _sentence_embeddings(self, sentences: List[str]) -> Optional[List[List[float]]]:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            return None

        model = SentenceTransformer("all-MiniLM-L6-v2")
        vectors = model.encode(sentences, normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]

    def _extract_entities(self, text: str) -> List[str]:
        emails = re.findall(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text)
        urls = re.findall(r"https?://\S+", text)
        dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
        proper = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)

        combined = emails + urls + dates + proper
        deduped: List[str] = []
        seen = set()
        for item in combined:
            if item.lower() in seen:
                continue
            seen.add(item.lower())
            deduped.append(item)
        return deduped[:12]

    def _extract_keywords(self, text: str, limit: int = 8) -> List[str]:
        words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
        filtered = [word for word in words if word not in STOPWORDS]
        if not filtered:
            return []
        counts = Counter(filtered)
        return [word for word, _ in counts.most_common(limit)]

    def _generate_summary(self, text: str) -> str:
        sentences = _split_sentences(text)
        if sentences:
            first = sentences[0]
            return first[:220] + ("..." if len(first) > 220 else "")

        keywords = self._extract_keywords(text, limit=5)
        if keywords:
            return "Key topics: " + ", ".join(keywords)
        return text[:220]

    def _build_parent_context(self, text: str) -> str:
        summary = self._generate_summary(text)
        return summary if len(summary) <= 300 else summary[:297] + "..."

    def _confidence_score(self, text: str) -> float:
        words = re.findall(r"\w+", text)
        if not words:
            return 0.0

        uniqueness = min(1.0, len(set(words)) / max(1, len(words)))
        length_signal = min(1.0, len(words) / 120.0)

        backend_bonus = {"semantic": 0.12, "recursive": 0.08}.get(self.backend, 0.05)
        score = 0.45 + (0.35 * uniqueness) + (0.15 * length_signal) + backend_bonus
        return round(min(0.99, max(0.2, score)), 3)


def _normalize_whitespace(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    return re.sub(r"[ \t]+", " ", cleaned).strip()


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _merge_short_chunks(chunks: List[str], minimum: int) -> List[str]:
    if not chunks:
        return []

    merged: List[str] = []
    buffer = ""
    for chunk in chunks:
        if len(chunk) < minimum:
            buffer = f"{buffer} {chunk}".strip()
            continue

        if buffer:
            merged.append(f"{buffer} {chunk}".strip())
            buffer = ""
        else:
            merged.append(chunk)

    if buffer:
        if merged:
            merged[-1] = f"{merged[-1]} {buffer}".strip()
        else:
            merged.append(buffer)

    return merged


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _chunk_metadata(chunk: EnrichedChunk) -> dict:
    data = asdict(chunk)
    data.pop("text", None)
    return data
