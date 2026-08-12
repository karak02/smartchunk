import unittest

from smartchunk import SmartChunk


SAMPLE_TEXT = (
    "SmartChunk helps Retrieval-Augmented Generation systems. "
    "It creates better context windows for language models. "
    "Contact support@smartchunk.dev for details. "
    "The next release is scheduled for 2026-09-15. "
    "Visit https://example.com/changelog for updates."
)


class TestSmartChunk(unittest.TestCase):
    def test_recursive_chunk_contains_required_fields(self):
        chunker = SmartChunk(backend="recursive", chunk_size=120, chunk_overlap=20)
        chunks = chunker.chunk(SAMPLE_TEXT)

        self.assertGreaterEqual(len(chunks), 1)
        first = chunks[0]
        for field in (
            "summary",
            "entities",
            "keywords",
            "parent_context",
            "prev_summary",
            "next_summary",
            "chunk_type",
            "confidence_score",
        ):
            self.assertTrue(hasattr(first, field), field)

        self.assertEqual(first.chunk_type, "recursive")
        self.assertGreaterEqual(first.confidence_score, 0.0)

    def test_fixed_size_backend(self):
        text = " ".join(["word"] * 500)
        chunker = SmartChunk(backend="fixed-size", chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(text)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.chunk_type == "fixed-size" for chunk in chunks))

    def test_export_methods(self):
        chunker = SmartChunk(backend="recursive", chunk_size=140)
        chunks = chunker.chunk(SAMPLE_TEXT)

        self.assertEqual(len(chunker.to_langchain(chunks)), len(chunks))
        self.assertEqual(len(chunker.to_llamaindex(chunks)), len(chunks))
        self.assertEqual(len(chunker.to_pinecone(chunks)), len(chunks))
        self.assertEqual(len(chunker.to_weaviate(chunks)), len(chunks))
        self.assertEqual(len(chunker.to_qdrant(chunks)), len(chunks))

        chroma = chunker.to_chroma(chunks)
        self.assertEqual(len(chroma["ids"]), len(chunks))
        self.assertEqual(len(chroma["documents"]), len(chunks))
        self.assertEqual(len(chroma["metadatas"]), len(chunks))

    def test_semantic_backend_fallback(self):
        chunker = SmartChunk(backend="semantic", chunk_size=140)
        chunks = chunker.chunk(SAMPLE_TEXT)

        self.assertGreaterEqual(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()
