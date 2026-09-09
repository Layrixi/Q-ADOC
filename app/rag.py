"""
rag.py

Takes the top-N reranked chunks from retriever and 
generates a natural-language answer, grounded in those chunks, using a local LLM via Ollama.

Model choice: for ANSWER GENERATION- Qwen2.5:7b is a good default here too - if you have the hardware, Qwen2.5:14b gives noticeably better answer quality atthe cost of speed.
72b is the best but it weights 47gb so nah im not downloading that, but you can
"""

import os
import requests
from typing import List

from app.retriever import Retriever

# ---------------------------------------------------------------------------
# Configuration

ANSWER_MODEL = "qwen2.5:14b" 
OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context documents. If the context does not contain enough "
    "information to answer, say so explicitly instead of guessing. Always "
    "mention which source(s) you used."
)


def call_local_llm(prompt: str, temperature: float = 0.25, max_tokens: int = 500) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": ANSWER_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def build_prompt(question: str, chunks: List[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(
            f"[Source {i}: {chunk['source']}]\n{chunk['text']}"
        )
    context_text = "\n\n".join(context_blocks)

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


class RAGPipeline:
    """
    Ties together retrieval and answer generation.
    """

    def __init__(self):
        self.retriever = Retriever()

    def answer(self, question: str, top_n: int = 5) -> dict:
        chunks = self.retriever.retrieve(question, stage2_top_n=top_n)

        if not chunks:
            return {
                "question": question,
                "answer": "No relevant information found in the indexed documents.",
                "sources": [],
            }

        prompt = build_prompt(question, chunks)
        answer_text = call_local_llm(prompt)

        return {
            "question": question,
            "answer": answer_text,
            "sources": [
                {"source": c["source"], "relevance_score": round(c["score"], 3)}
                for c in chunks
            ],
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, help="Question to ask the RAG system")
    parser.add_argument("--top_n", type=int, default=5)
    args = parser.parse_args()

    pipeline = RAGPipeline()
    result = pipeline.answer(args.question, top_n=args.top_n)

    print(f"\nQ: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print("Sources used:")
    for s in result["sources"]:
        print(f"  - {s['source']} (relevance: {s['relevance_score']})")
