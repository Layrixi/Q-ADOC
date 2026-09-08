"""
retriever.py

Two-stage retrieval pipeline

Stage 1 (fast, broad):  retrieves top-K candidates from potentially thousands of chunks, cheaply.
Stage 2 (slow, precise): fine-tuned reranker re-scores those K candidates and picks the best top-N to hand to the LLM.

"""

import os
import json
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

# ---------------------------------------------------------------------------
# Configuration

CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
COLLECTION_NAME = "documents"

# Bi-encoder for stage 1, small, fast, and good enough for stage-1 recall
BI_ENCODER_MODEL = "all-MiniLM-L6-v2"

# fine-tuned cross-encoder 
# Falls back to the base (non fine-tuned) model if the fine-tuned one isn't found so this module still works before you've run training.
FINETUNED_RERANKER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "reranker-finetuned")
BASE_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

STAGE1_TOP_K = 25   # how many candidates the bi-encoder retrieves
STAGE2_TOP_N = 5    # how many reranker keeps for the LLM


# ---------------------------------------------------------------------------
# Indexing

def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)

def get_or_create_collection(client):
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=BI_ENCODER_MODEL
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

def index_chunks(chunks_path: str):
    """
    Reads chunks produced by app/ingest.py (data/chunks.jsonl) and indexes them into ChromaDB. 
    """
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"{chunks_path} not found"
        )

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    ids, documents, metadatas = [], [], []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            ids.append(chunk["id"])
            documents.append(chunk["text"])
            metadatas.append({"source": chunk.get("source", "unknown")})

    if not ids:
        print("No chunks found in file - nothing to index.")
        return

    # Chroma has a batch size limit on some 
    BATCH = 500
    for i in range(0, len(ids), BATCH):
        collection.upsert(
            ids=ids[i:i + BATCH],
            documents=documents[i:i + BATCH],
            metadatas=metadatas[i:i + BATCH],
        )

    print(f"Indexed {len(ids)} chunks into ChromaDB collection '{COLLECTION_NAME}'.")


# ---------------------------------------------------------------------------
# Retrieval

class Retriever:
    """
    Wraps stage-1 (ChromaDB bi-encoder search) and stage-2 (cross-encoder
    reranking) into a single .retrieve(question) call.

    Loading the reranker model happens once, in __init__, since loading
    model weights on every query would be slow.
    """

    def __init__(self, reranker_path: Optional[str] = None):
        self.client = get_chroma_client()
        self.collection = get_or_create_collection(self.client)

        # Prefer the fine-tuned reranker, fall back to the base model with a warning if it hasn't been trained yet.
        path_to_load = reranker_path or FINETUNED_RERANKER_PATH
        if os.path.isdir(path_to_load) and os.listdir(path_to_load):
            print(f"Loading fine-tuned reranker from {path_to_load}")
            self.reranker = CrossEncoder(path_to_load)
        else:
            print(
                f"[INFO] No fine-tuned reranker found at {path_to_load}. "
                f"Falling back to base model: {BASE_RERANKER_MODEL}. "
                f"Run training to produce a fine-tuned one."
            )
            self.reranker = CrossEncoder(BASE_RERANKER_MODEL)

    def retrieve(self, question: str, stage1_top_k: int = STAGE1_TOP_K, stage2_top_n: int = STAGE2_TOP_N) -> List[dict]:
        """
        Returns the top-N chunks (dicts with 'text', 'source', 'score') most relevant to the question, after two-stage retrieval + reranking.
        """
        # --- Stage 1: bi-encoder candidate retrieval ---
        results = self.collection.query(
            query_texts=[question],
            n_results=stage1_top_k,
        )

        candidate_texts = results["documents"][0]
        candidate_metadatas = results["metadatas"][0]

        if not candidate_texts:
            return []

        # --- Stage 2: cross-encoder reranking ---
        pairs = [[question, text] for text in candidate_texts]
        scores = self.reranker.predict(pairs)

        reranked = sorted(
            zip(candidate_texts, candidate_metadatas, scores),
            key=lambda x: x[2],
            reverse=True,
        )

        top_results = reranked[:stage2_top_n]

        return [
            {
                "text": text,
                "source": meta.get("source", "unknown"),
                "score": float(score),
            }
            for text, meta, score in top_results
        ]


# ---------------------------------------------------------------------------
# CLI entry point - useful for quick manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        action="store_true",
        help="(Re)index chunks from data/chunks.jsonl into ChromaDB.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single test query against the indexed collection.",
    )
    args = parser.parse_args()

    if args.index:
        chunks_path = os.path.join(os.path.dirname(__file__), "..", "data", "chunks.jsonl")
        index_chunks(chunks_path)

    if args.query:
        retriever = Retriever()
        results = retriever.retrieve(args.query)
        print(f"\nTop results for: {args.query}\n")
        for i, r in enumerate(results, 1):
            if i == 1:
                print(f"[{i}] score={r['score']:.3f} source={r['source']}")
                print(f"    {r['text']}...\n")
            else:
                print(f"[{i}] score={r['score']:.3f} source={r['source']}")
                print(f"    {r['text'][:200]}...\n") 
