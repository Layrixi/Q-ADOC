"""
Synthetic dataset generator for reranker. For each chunk LLM generates a question, for which this chunk is the anwser.
Creates positive (labeled 1) and negative (labeled 0) pairs that are saved to jsonl file.
May be improved using real Q&A chunks, but they are unavailable to me(or couldn't find them yet)
Model used for generation: Ollama, qwen2.5 7b

Make sure ollama is running in the background before launching, otherwise Ollama will not anwser :<

Uses hard-negative mining, so some neg chunks are more similiar to positive to make it harder for the model
Splits into hard negatives (similiar ones)
And random negatives (unrelated random ones)


"""

import json
import random
import os
import requests
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer, util


QUESTION_GEN_MODEL = "qwen2.5:7b-instruct"
OLLAMA_URL = "http://localhost:11434/api/generate" # using default port
NUM_NEGATIVES_PER_POSITIVE = 3       # n negative per positive
RANDOM_SEED = 67
MIN_WORDS_FOR_QUESTION_GEN = 15      # to skip headlines or tables with no context

#parallel llm calls. If you, the one who downloaded the repo, use it, make sure your gpu can handle it
MAX_WORKERS = 2


# --- Hard-negative mining config ---
BI_ENCODER_MODEL = "all-MiniLM-L6-v2"   # same model used in retriever later to reflect what it would surface as false-positive
HARD_NEGATIVE_RATIO = 0.7               # fraction of negatives that should be "hard" vs fully random
HARD_NEGATIVE_POOL_SIZE = 10            # sample hard negatives from top-N similar candidates (adds diversity)
HARD_NEGATIVE_SIM_MIN = 0.40            # below this: not similar enough to be "hard"
HARD_NEGATIVE_SIM_MAX = 0.90            # above this: too similar(risk of overlapping)
 
_embedder_cache = None
 
 
def get_embedder() -> SentenceTransformer:
    """Loads the bi-encoder once and caches it."""
    global _embedder_cache
    if _embedder_cache is None:
        _embedder_cache = SentenceTransformer(BI_ENCODER_MODEL)
    return _embedder_cache

def call_local_llm(prompt: str, temperature: float = 0.67, max_tokens: int = 100) -> str:
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

#this can be ran in parallel
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

    # Strip of "question:", prefixes etc just in case
    question = question.strip().strip('"').strip()
    if question.lower().startswith("question:"):
        question = question[len("question:"):].strip()

    return question

#parallelism
def generate_questions_concurrently(
    valid_chunks: List[dict], max_workers: int = MAX_WORKERS
) -> Dict[str, Optional[str]]:
    """
    Generates questions for multiple chunks at once, using a thread pool to
    fire several requests at Ollama concurrently instead of waiting for each
    one to finish before starting the next.
 
    Returns a dict mapping chunk_id -> question (or None if that chunk's generation failed - failures are logged but don't stop the rest).
    """
 
    def _generate_one(chunk: dict):
        try:
            return chunk["id"], generate_question_for_chunk(chunk["text"])
        except Exception as e:
            print(f"[WARN] Couldn't generate question for {chunk['id']}: {e}")
            return chunk["id"], None
 
    results: Dict[str, Optional[str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_generate_one, chunk) for chunk in valid_chunks]
        completed = 0
        for future in as_completed(futures):
            chunk_id, question = future.result()
            results[chunk_id] = question
            completed += 1
            if completed % 20 == 0:
                print(f"Generated {completed}/{len(valid_chunks)} questions...")
 
    return results

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

def find_hard_negative_candidates(
    question_embedding,
    chunk_embeddings,
    all_chunk_ids: List[str],
    exclude_id: str,
    pool_size: int = HARD_NEGATIVE_POOL_SIZE,
    sim_min: float = HARD_NEGATIVE_SIM_MIN,
    sim_max: float = HARD_NEGATIVE_SIM_MAX,
) -> List[str]:
    """
    Returns up to `pool_size` chunk ids whose similarity to the question falls within [sim_min, sim_max]
    """
    similarities = util.cos_sim(question_embedding, chunk_embeddings)[0]
 
    candidates = []
    for idx, cid in enumerate(all_chunk_ids):
        if cid == exclude_id:
            continue
        sim = similarities[idx].item()
        if sim_min <= sim <= sim_max:
            candidates.append((cid, sim))
 
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in candidates[:pool_size]]
 

def build_dataset(
        chunks: List[dict], 
        num_negatives: int = NUM_NEGATIVES_PER_POSITIVE,
        hard_negative_ratio: float = HARD_NEGATIVE_RATIO,
        ) -> List[TrainingExample]:
    """
    chunks: list of dicts {"id": str, "text": str}

    Returns list TrainingExample for further saving
    """
    random.seed(RANDOM_SEED)
    examples: List[TrainingExample] = []

    chunk_ids = [c["id"] for c in chunks]            # mutable - used by the random-negative 
    all_chunk_ids = [c["id"] for c in chunks]        # static, aligned index-for-index with chunk_embeddings
    chunk_by_id = {c["id"]: c["text"] for c in chunks}
 
    print("___________Embedding all chunks for hard-negative mining_________")
    embedder = get_embedder()
    chunk_embeddings = embedder.encode(
        [c["text"] for c in chunks], convert_to_tensor=True, show_progress_bar=True
    )
 
    # Pass 1: filter out chunks that can't produce a sensible question
    valid_chunks = [
        c for c in chunks
        if len(c["text"].split()) >= MIN_WORDS_FOR_QUESTION_GEN
        and not looks_like_table_or_stats(c["text"])
    ]
    print(f"{len(valid_chunks)}/{len(chunks)} chunks are eligible for question generation.")
 
    # generate all questions concurrently (the slow, LLM-bound part) 
    print(f"Generating questions with {MAX_WORKERS} concurrent workers...")
    question_by_chunk_id = generate_questions_concurrently(valid_chunks, max_workers=MAX_WORKERS)
 
    # build positive/negative pairs  
    for chunk_index, chunk in enumerate(chunks):
        chunk_id = chunk["id"]
        chunk_text = chunk["text"]
 
        question = question_by_chunk_id.get(chunk_id)
        if question is None:
            # either this chunk was filtered out in pass 1, or generation failed
            continue
 
        # positive
        examples.append(TrainingExample(
            question=question,
            chunk_text=chunk_text,
            label=1,
            source_chunk_id=chunk_id,
        ))
 
        # hard negatives, chunks semantically similar to the question but wrong
        question_embedding = embedder.encode(question, convert_to_tensor=True)
        hard_candidates = find_hard_negative_candidates(
            question_embedding, chunk_embeddings, all_chunk_ids, exclude_id=chunk_id
        )
        num_hard_wanted = round(num_negatives * hard_negative_ratio)
        hard_negative_ids = random.sample(hard_candidates, k=min(num_hard_wanted, len(hard_candidates)))
 
        num_random_needed = num_negatives - len(hard_negative_ids)
 
        # remaining negatives, fully random
        # Temporarily swap chunk_id to the end and pop it, restores the original order after.
        last_index = len(chunk_ids) - 1
        chunk_ids[chunk_index], chunk_ids[last_index] = chunk_ids[last_index], chunk_ids[chunk_index]
        chunk_ids.pop()
 
        random_negative_ids = []
        if num_random_needed > 0 and len(chunk_ids) > 0:
            # sample a few extra in case some collide with already-chosen hard negatives
            sample_pool = random.sample(
                chunk_ids, k=min(num_random_needed + len(hard_negative_ids), len(chunk_ids))
            )
            random_negative_ids = [cid for cid in sample_pool if cid not in hard_negative_ids][:num_random_needed]
 
        chunk_ids.append(chunk_id)
        chunk_ids[chunk_index], chunk_ids[last_index] = chunk_ids[last_index], chunk_ids[chunk_index]
 
        for neg_id in hard_negative_ids + random_negative_ids:
            examples.append(TrainingExample(
                question=question,
                chunk_text=chunk_by_id[neg_id],
                label=0,
                source_chunk_id=neg_id,
            ))
 
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
