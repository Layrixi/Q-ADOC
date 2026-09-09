"""
main.py

FastAPI backend for the RAG application. Exposes:

  POST /upload       - upload a document (PDF/DOCX/TXT), ingest + index it
  POST /ask          - ask a question, get an answer grounded in indexed docs
  GET  /documents    - list currently indexed document sources
  GET  /health       - basic health check

Run with:
    uvicorn app.main:app --reload --port 8000
"""

import os
import shutil
import tempfile
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.ingest import ingest_document, save_chunks
from app.retriever import index_chunks, get_chroma_client, get_or_create_collection, COLLECTION_NAME
from app.rag import RAGPipeline

app = FastAPI(title="Document Q&A (RAG + fine-tuned reranker)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chunks.jsonl")

# The RAG pipeline loads model weights (bi-encoder, reranker) once at
# startup rather than per-request, since loading them is the slow part.
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


class AskRequest(BaseModel):
    question: str
    top_n: int = 5


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts a PDF/DOCX/TXT file, chunks it, appends the chunks to
    data/chunks.jsonl, and re-indexes into ChromaDB.

    Note: this appends to the existing chunk store rather than replacing it,
    so you can upload multiple documents incrementally.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".pdf", ".docx", ".txt"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        new_chunks = ingest_document(tmp_path)
        # Re-tag chunks with the original filename (tempfile has a random name)
        for chunk in new_chunks:
            chunk["source"] = file.filename
    finally:
        os.remove(tmp_path)

    if not new_chunks:
        raise HTTPException(status_code=422, detail="No extractable text found in document.")

    # Append to existing chunks file
    os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)
    file_exists = os.path.exists(CHUNKS_PATH)
    mode = "a" if file_exists else "w"
    import json
    with open(CHUNKS_PATH, mode, encoding="utf-8") as f:
        for chunk in new_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Re-index just the new chunks into the existing ChromaDB collection
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    collection.upsert(
        ids=[c["id"] for c in new_chunks],
        documents=[c["text"] for c in new_chunks],
        metadatas=[{"source": c["source"]} for c in new_chunks],
    )

    return {
        "filename": file.filename,
        "chunks_added": len(new_chunks),
        "message": "Document indexed successfully.",
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    pipeline = get_pipeline()
    result = pipeline.answer(request.question, top_n=request.top_n)
    return result


@app.get("/documents")
def list_documents():
    """Returns the distinct document sources currently indexed."""
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    all_items = collection.get()
    sources = set()
    for meta in all_items.get("metadatas", []):
        if meta and "source" in meta:
            sources.add(meta["source"])

    return {"documents": sorted(sources), "total_chunks": len(all_items.get("ids", []))}
