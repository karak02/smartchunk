import os
import sys
import time
import shutil
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add src/ to path so we can import smartchunk directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smartchunk import SmartChunker
from smartchunk.models import ChunkStrategy, EnrichmentField

import smartchunk
print("DEBUG: SMARTCHUNK IMPORTED FROM:", smartchunk.__file__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smartchunk-demo")

app = FastAPI(title="SmartChunk Demo App")

# Create temporary upload directory
TEMP_DIR = Path(__file__).parent / "temp"
TEMP_DIR.mkdir(exist_ok=True)

class TextProcessRequest(BaseModel):
    text: str
    source: str = "raw_input.txt"
    strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 50
    enrich: bool = True
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    enrichments: List[str] = ["summary", "entities", "keywords", "confidence"]

def get_chunker_instance(
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    enrich: bool,
    model: str,
    temperature: float,
    enrichments: List[str]
) -> SmartChunker:
    """Helper to construct a SmartChunker instance with explicit config."""
    try:
        strat = ChunkStrategy(strategy.lower())
    except ValueError:
        strat = ChunkStrategy.RECURSIVE

    field_enums = []
    for f in enrichments:
        try:
            field_enums.append(EnrichmentField(f.lower()))
        except ValueError:
            pass

    if not field_enums:
        field_enums = list(EnrichmentField)

    return SmartChunker(
        model=model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strat,
        enrich=enrich,
        enrichments=field_enums,
        temperature=temperature,
    )

@app.post("/api/process/text")
async def process_text(req: TextProcessRequest):
    """Process raw text into SmartChunks."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text content cannot be empty")

    logger.info("Processing text via API: strategy=%s, model=%s, enrich=%s", req.strategy, req.model, req.enrich)
    
    start_time = time.time()
    try:
        chunker = get_chunker_instance(
            strategy=req.strategy,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            enrich=req.enrich,
            model=req.model,
            temperature=req.temperature,
            enrichments=req.enrichments
        )

        chunks = chunker.process_text(req.text, source=req.source)
        duration = time.time() - start_time
        
        serialized_chunks = [chunk.model_dump() for chunk in chunks]
        
        return {
            "success": True,
            "chunks": serialized_chunks,
            "stats": {
                "total_chunks": len(chunks),
                "duration_seconds": round(duration, 3),
                "usage": chunker.usage_stats,
                "cache": chunker.cache_stats
            }
        }
    except Exception as e:
        logger.exception("Failed processing text")
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": str(e)}
        )

from litellm import acompletion
from smartchunk.models import SmartChunk, RetrievalConfig
from smartchunk.retrieval import HybridRetriever

class QueryRequest(BaseModel):
    query: str
    chunks: list[dict]
    model: str = "gpt-4o-mini"

@app.post("/api/query")
async def query_document(req: QueryRequest):
    """Query document using SmartChunk vs. Traditional Chunk retrieval side-by-side."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if not req.chunks:
        raise HTTPException(status_code=400, detail="Document chunks list is empty")

    logger.info("Executing retrieval comparison for query: %s", req.query)

    try:
        # Reconstruct SmartChunks
        smart_chunks = []
        for c in req.chunks:
            smart_chunks.append(SmartChunk.model_validate(c))

        # 1. SmartChunk Retrieval (Hybrid search with contextual text and graph expansion)
        smart_config = RetrievalConfig(
            vector_weight=0.5,
            bm25_weight=0.5,
            expand_parents=True,
            expand_neighbors=1,
            reranker_model=None
        )
        smart_retriever = HybridRetriever(smart_chunks, config=smart_config)
        smart_results = smart_retriever.search(req.query, top_k=3)

        # 2. Traditional Retrieval (Dense/BM25 on raw text without context or expansion)
        traditional_chunks = []
        for c in smart_chunks:
            c_copy = c.model_copy(deep=True)
            c_copy.contextual_text = ""  # Strip contextual embedding payload
            traditional_chunks.append(c_copy)

        trad_config = RetrievalConfig(
            vector_weight=0.5,
            bm25_weight=0.5,
            expand_parents=False,
            expand_neighbors=0,
            reranker_model=None
        )
        trad_retriever = HybridRetriever(traditional_chunks, config=trad_config)
        traditional_results = trad_retriever.search(req.query, top_k=3)

        # 3. LLM Generation: Answer query based on SmartChunk expanded context
        context_parts = []
        for idx, res in enumerate(smart_results):
            context_parts.append(
                f"Chunk #{idx+1} (Source: {res.chunk.metadata.source} | Section: {res.chunk.parent_context})\n"
                f"Text: {res.chunk.text}"
            )
        context_str = "\n\n---\n\n".join(context_parts)
        
        prompt = (
            "You are an assistant answering questions about a document using the provided retrieved context.\n"
            "Answer the query accurately, concisely, and based ONLY on the provided context.\n"
            "If the answer cannot be found in the context, say 'I cannot find the answer in the retrieved context.'\n\n"
            f"Context:\n{context_str}\n\n"
            f"Query: {req.query}\n\n"
            "Answer:"
        )

        try:
            response = await acompletion(
                model=req.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            answer = response.choices[0].message.content.strip()
        except Exception as llm_err:
            logger.warning("LLM generation failed: %s", llm_err)
            answer = "Error generating LLM answer: " + str(llm_err)

        # Serialize results for JSON response
        serialized_smart = []
        for r in smart_results:
            serialized_smart.append({
                "chunk": r.chunk.model_dump(),
                "score": round(r.score, 3),
                "bm25_score": round(r.bm25_score, 3) if r.bm25_score is not None else 0.0,
                "dense_score": round(r.dense_score, 3) if r.dense_score is not None else 0.0,
            })

        serialized_trad = []
        for r in traditional_results:
            serialized_trad.append({
                "chunk": r.chunk.model_dump(),
                "score": round(r.score, 3),
            })

        return {
            "success": True,
            "answer": answer,
            "smart_results": serialized_smart,
            "traditional_results": serialized_trad
        }
    except Exception as e:
        logger.exception("Failed processing query")
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": str(e)}
        )

@app.post("/api/process/file")
async def process_file(
    file: UploadFile = File(...),
    strategy: str = Form("recursive"),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(50),
    enrich: bool = Form(True),
    model: str = Form("gpt-4o-mini"),
    temperature: float = Form(0.0),
    enrichments: str = Form("summary,entities,keywords,confidence")
):
    """Process an uploaded document (PDF, TXT, MD, DOCX, HTML) into SmartChunks."""
    logger.info("Processing file upload via API: filename=%s, strategy=%s, model=%s", file.filename, strategy, model)
    
    temp_file_path = TEMP_DIR / file.filename
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    enrichments_list = [f.strip() for f in enrichments.split(",") if f.strip()]

    start_time = time.time()
    try:
        chunker = get_chunker_instance(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            enrich=enrich,
            model=model,
            temperature=temperature,
            enrichments=enrichments_list
        )

        chunks = chunker.process(temp_file_path)
        duration = time.time() - start_time
        
        serialized_chunks = [chunk.model_dump() for chunk in chunks]
        
        return {
            "success": True,
            "chunks": serialized_chunks,
            "stats": {
                "total_chunks": len(chunks),
                "duration_seconds": round(duration, 3),
                "usage": chunker.usage_stats,
                "cache": chunker.cache_stats
            }
        }
    except Exception as e:
        logger.exception("Failed processing file upload")
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": str(e)}
        )
    finally:
        if temp_file_path.exists():
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.error("Failed to delete temp file %s: %s", temp_file_path, e)

# Serve static dashboard files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Mount static files (HTML, CSS, JS)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
