"""
train_reranker.py

Fine-tuning code for the documents I'd like it to be fine-tuned to. Using ms-marco-MiniLM-L-6 coz I need to be resource efficient, you can change it if you are fine-tuning it for your own usage
-- fire a smoke test first (passed as an argument) to check if everything works
"""

import argparse
import json
import os
import random
from typing import List, Tuple

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator
from torch.utils.data import DataLoader
from sentence_transformers import InputExample

# ---------------------------------------------------------------------------
# config

BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

NUM_EPOCHS = 4
BATCH_SIZE = 8
LEARNING_RATE = 2.8749194736980314e-05
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.07577274065180203
VAL_SPLIT_RATIO = 0.15

RANDOM_SEED = 67

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "training", "data", "training_data.jsonl")
OUTPUT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "reranker-finetuned")
METRICS_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "training_metrics.json")


# ---------------------------------------------------------------------------
# Load and data split

def load_training_data(
        path: str) -> List[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def split_train_val(
    examples: List[dict], 
    val_ratio: float, 
    seed: int
) -> Tuple[List[dict], List[dict]]:
    """
    splits by source_chunk_id so the same question with positive and negative doesn't go to train/val at the same time. (data leakage thing)
    """
    questions = list(set(ex["question"] for ex in examples))
    random.Random(seed).shuffle(questions)

    val_size = int(len(questions) * val_ratio)
    val_questions = set(questions[:val_size])

    train_examples = [ex for ex in examples if ex["question"] not in val_questions]
    val_examples = [ex for ex in examples if ex["question"] in val_questions]

    return train_examples, val_examples


def to_input_examples(
        examples: List[dict]) -> List[InputExample]:
    return [
        InputExample(texts=[ex["question"], ex["chunk_text"]], label=float(ex["label"]))
        for ex in examples
    ]


# ---------------------------------------------------------------------------
# Evaluation

def evaluate_model(
        model: CrossEncoder, 
        val_examples: List[dict], 
        name: str) -> dict:
    """
    uses built in evaluator to, well, evaluate the model
    """
    sentence_pairs = [[ex["question"], ex["chunk_text"]] for ex in val_examples]
    labels = [ex["label"] for ex in val_examples]

    evaluator = CrossEncoderClassificationEvaluator(
        sentence_pairs, labels, name=name
    )
    results = evaluator(model)
    return results


# ---------------------------------------------------------------------------
# Main training 

def run_training(
        smoke_test: bool = False
        ):

    random.seed(RANDOM_SEED)

    if not os.path.exists(DATA_PATH):
        print(f"file missing at: {DATA_PATH}")
        return

    print("Loading training data...")
    all_examples = load_training_data(DATA_PATH)
    print(f"Loaded {len(all_examples)} pairs (question, chunk).")

    if smoke_test:
        # to check if it actually improves the model
        # testing on limited num of examples and 2 epochs to make sure lr is 'warmed up'
        print("\n[SMOKE TEST]\n")
        random.shuffle(all_examples)
        all_examples = all_examples[:1500]
        epochs = 2
    else:
        epochs = NUM_EPOCHS

    train_examples, val_examples = split_train_val(all_examples, VAL_SPLIT_RATIO, RANDOM_SEED)
    print(f"SPLIT: {len(train_examples)} train / {len(val_examples)} val")

    # Evaluate before fine-tuning
    print(f"\n base model loading: {BASE_MODEL}")
    model = CrossEncoder(BASE_MODEL, num_labels=1)

    print("Evaluating before fine-tuning")
    baseline_metrics = evaluate_model(model, val_examples, name="baseline")
    print(f"Baseline metrics: {baseline_metrics}")

    # ---- Fine-tuning ----
    train_dataloader = DataLoader(
        to_input_examples(train_examples), shuffle=True, batch_size=BATCH_SIZE
    )
    warmup_steps = int(len(train_dataloader) * epochs * WARMUP_RATIO)

    print(
        f"\nfine-tuning: {epochs} epochs, batch size={BATCH_SIZE}, "
        f"lr={LEARNING_RATE}, warmup steps={warmup_steps}"
    )
    model.fit(
        train_dataloader=train_dataloader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
        show_progress_bar=True,
    )

    # ---- eval after fine-tuning with same data as for baseline eval ----
    print("\nEvaluating after fine-tuning")
    finetuned_metrics = evaluate_model(model, val_examples, name="finetuned")
    print(f"Fine-tuned metrics: {finetuned_metrics}")

    # ---- save model and metrics ----
    if not smoke_test:
        #model save
        os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
        model.save(OUTPUT_MODEL_DIR)
        print(f"\nsaved model to: {OUTPUT_MODEL_DIR}")
        #metrics save
        os.makedirs(os.path.dirname(METRICS_OUTPUT_PATH), exist_ok=True)
        with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "baseline": baseline_metrics,
                    "finetuned": finetuned_metrics,
                    "config": {
                        "base_model": BASE_MODEL,
                        "num_epochs": epochs,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "train_size": len(train_examples),
                        "val_size": len(val_examples),
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"metrics saved to: {METRICS_OUTPUT_PATH}")
    else:
        print("\n[SMOKE TEST] no save")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Starts a quick test on a small data batch to check the metrics",
    )
    args = parser.parse_args()

    run_training(smoke_test=args.smoke_test)
