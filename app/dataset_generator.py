"""
Synthetic dataset generator for reranker. For each chunk LLM generates a question, for which this chunk is the anwser.
Creates positive (labeled 1) and negative (labeled 0) pairs that are saved to jsonl file.
May be improved using real Q&A chunks, but they are unavailable to me(or couldn't find them yet)
Model used for generation: Ollama, qwen2.5 7b

Make sure ollama is running in the background before launching, otherwise Ollama will not anwser :<

"""

import json
import random
import os
import requests
from dataclasses import dataclass, asdict
from typing import List


QUESTION_GEN_MODEL = "qwen2.5:7b-instruct"
OLLAMA_URL = "http://localhost:11434/api/generate" # using default port
NUM_NEGATIVES_PER_POSITIVE = 3       # n negative per positive
RANDOM_SEED = 67
MIN_WORDS_FOR_QUESTION_GEN = 15      # to skip headlines or tables with no context


def call_local_llm(prompt: str, temperature: float = 0.7, max_tokens: int = 100) -> str:
    """
    calls ollama with REST api.
    """
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": QUESTION_GEN_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


@dataclass
class TrainingExample:
    question: str
    chunk_text: str
    label: int  
    source_chunk_id: str

#think this could be optimized?
def generate_question_for_chunk(chunk_text: str) -> str:
    prompt = (
        "Below is a fragment of a document (a tabletop RPG rulebook or "
        "sourcebook). Come up with ONE specific question that this fragment "
        "directly and fully answers. The question should sound like something "
        "a player or game master would actually type into a search box. "
        "Return ONLY the question, nothing else - no preamble, no quotes.\n\n"
        f"Fragment:\n{chunk_text}\n\nQuestion:"
    )

    question = call_local_llm(prompt, temperature=0.7, max_tokens=100)

    # Strip of "question:", prefixes etc
    question = question.strip().strip('"').strip()
    if question.lower().startswith("question:"):
        question = question[len("question:"):].strip()

    return question


def looks_like_table_or_stats(chunk_text: str) -> bool:
    """
    Table/stats filter. 
    Lot of short words, numbers, symbols, no full sentences = filter out, since it may hallucinate 
    """
    words = chunk_text.split()
    if len(words) == 0:
        return True

    num_short_tokens = sum(1 for w in words if len(w) <= 3)
    num_sentences = chunk_text.count(".") + chunk_text.count("!") + chunk_text.count("?")

    short_token_ratio = num_short_tokens / len(words)
    avg_words_per_sentence = len(words) / max(num_sentences, 1)

    # compute ratio of !>?. to words. Heuristic, may need tuning to correctly assume if it's a sentence or a statblock/table
    return short_token_ratio > 0.45 and avg_words_per_sentence > 40


def build_dataset(chunks: List[dict], num_negatives: int = NUM_NEGATIVES_PER_POSITIVE) -> List[TrainingExample]:
    """
    chunks: list of dicts {"id": str, "text": str}

    Returns list TrainingExample for further saving
    """
    random.seed(RANDOM_SEED)
    examples: List[TrainingExample] = []

    chunk_ids = [c["id"] for c in chunks]
    chunk_by_id = {c["id"]: c["text"] for c in chunks}

    for chunk_index, chunk in enumerate(chunks):
        chunk_id = chunk["id"]
        chunk_text = chunk["text"]

        # skip short chunks since they can't generate anything reasonable
        # + skip stat blocks / tables
        # split used twice, optimize later
        if len(chunk_text.split()) < MIN_WORDS_FOR_QUESTION_GEN or looks_like_table_or_stats(chunk_text):
            continue

        try:
            question = generate_question_for_chunk(chunk_text)
        except Exception as e:
            print(f"[WARN] Couldn't generate question for {chunk_id} | {chunk_text}: {e}")
            continue

        # positive
        examples.append(TrainingExample(
            question=question,
            chunk_text=chunk_text,
            label=1,
            source_chunk_id=chunk_id,
        ))

        # Negative (may generate positive sometimes),random chunk anwsers the question.
        # Temporarily swap chunk_id to the end and pop it\
        # restores the original order after.
        last_index = len(chunk_ids) - 1
        chunk_ids[chunk_index], chunk_ids[last_index] = chunk_ids[last_index], chunk_ids[chunk_index]
        chunk_ids.pop()

        if len(chunk_ids) == 0:
            chunk_ids.append(chunk_id)
            continue

        negative_ids = random.sample(chunk_ids, k=min(num_negatives, len(chunk_ids)))
        for neg_id in negative_ids:
            examples.append(TrainingExample(
                question=question,
                chunk_text=chunk_by_id[neg_id],
                label=0,
                source_chunk_id=neg_id,
            ))

        chunk_ids.append(chunk_id)
        chunk_ids[chunk_index], chunk_ids[last_index] = chunk_ids[last_index], chunk_ids[chunk_index]

    return examples


def save_dataset(examples: List[TrainingExample], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")
    print(f"saved {len(examples)} examples to {output_path}")


def load_chunks_from_jsonl(path: str) -> List[dict]:
    """loads chunks from path variable (app/ingest.py)"""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


if __name__ == "__main__":

    chunks_path = os.path.join(os.path.dirname(__file__), "..", "training", "data", "chunks.jsonl")
    output_path = os.path.join(os.path.dirname(__file__), "..", "training", "data", "training_data.jsonl")

    if not os.path.exists(chunks_path):
        print(f"{chunks_path} missing.")
        exit(1)

    chunks = load_chunks_from_jsonl(chunks_path)
    print(f"Loaded {len(chunks)} chunks.")
    print("Generating questions for fine-tuning...")

    examples = build_dataset(chunks)
    save_dataset(examples, output_path)
