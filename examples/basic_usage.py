from smartchunk import SmartChunk


if __name__ == "__main__":
    text = (
        "SmartChunk enriches chunks with metadata. "
        "Use it for better retrieval and response grounding in RAG applications. "
        "It also exports records for common vector stores."
    )

    chunker = SmartChunk(backend="recursive", chunk_size=110)
    chunks = chunker.chunk(text)

    print("Chunks generated:", len(chunks))
    for i, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {i}: {chunk.text}")
        print("Summary:", chunk.summary)
        print("Keywords:", chunk.keywords)

    print("\nLangChain format sample:")
    print(chunker.to_langchain(chunks)[0])
