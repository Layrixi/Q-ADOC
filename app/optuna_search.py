"""
hyperparameter_search.py

Uses Optuna to search the reranker's hyperparameters, on a reduced subset of
the data (default 10%) so the search is feasible on limited hardware.

Searched parameters:
  - learning_rate   (1e-5 to 5e-5, log scale - typical for learning rates,
                     since the effect of going from 1e-5 to 2e-5 is
                     proportionally similar to 2e-5 to 4e-5)
  - num_epochs      (2 to 4)
  - warmup_ratio    (0.0 to 0.2) - how gradually the learning rate ramps up
                     at the start of training
  - weight_decay    (0.0 or 0.01) - regularization strength
  - batch_size      (8, 16, 32, or 64)

also:
  - Models are NEVER saved by this script - only metrics + configs, to
    ../models/hyperparameter_search_results.json
  - A trial that crashes (e.g. CUDA out-of-memory on a large batch_size) is
    caught, logged as failed, and the search continues with the next trial.
  - Results are written to disk after EVERY trial, not just at the end.
"""

import gc
import json
import os
import random
import time
from typing import List, Tuple

import optuna
import torch
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Config

BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RANDOM_SEED = 67
DATA_FRACTION = 0.1     
VAL_SPLIT_RATIO = 0.15
N_TRIALS = 30

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "training", "data", "training_data.jsonl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "hyperparameter_search_results.json")

TARGET_METRIC_SUFFIX = "average_precision"

# Module-level data, set once in run_search() and read inside objective().
train_examples: List[dict] = []
val_examples: List[dict] = []


# ---------------------------------------------------------------------------
# Data loading / splitting (same logic as train_reranker.py)

def load_training_data(path: str) -> List[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def split_train_val(
    examples: List[dict], val_ratio: float, seed: int
) -> Tuple[List[dict], List[dict]]:
    """
    Splits by question (not by row) so the same question's positive/negative
    pairs don't end up split across train and val. avoids data leakage.
    """
    questions = list(set(ex["question"] for ex in examples))
    random.Random(seed).shuffle(questions)

    val_size = int(len(questions) * val_ratio)
    val_questions = set(questions[:val_size])

    train = [ex for ex in examples if ex["question"] not in val_questions]
    val = [ex for ex in examples if ex["question"] in val_questions]

    return train, val


def to_input_examples(examples: List[dict]) -> List[InputExample]:
    return [
        InputExample(texts=[ex["question"], ex["chunk_text"]], label=float(ex["label"]))
        for ex in examples
    ]


def evaluate_model(model: CrossEncoder, examples: List[dict], name: str) -> dict:
    sentence_pairs = [[ex["question"], ex["chunk_text"]] for ex in examples]
    labels = [ex["label"] for ex in examples]
    evaluator = CrossEncoderClassificationEvaluator(sentence_pairs, labels, name=name)
    return evaluator(model)


def get_target_metric(metrics: dict, name: str) -> float:
    return metrics[f"{name}_{TARGET_METRIC_SUFFIX}"]


def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Optuna objective

def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    epochs = trial.suggest_int("num_epochs", 2, 4)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    weight_decay = trial.suggest_categorical("weight_decay", [0.0, 0.01])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])

    print(f"\n--- Trial {trial.number}: lr={lr:.2e}, epochs={epochs}, "
          f"warmup={warmup_ratio:.2f}, weight_decay={weight_decay}, batch_size={batch_size} ---")

    start_time = time.time()
    model = None

    try:
        model = CrossEncoder(BASE_MODEL, num_labels=1)

        train_dataloader = DataLoader(
            to_input_examples(train_examples), shuffle=True, batch_size=batch_size
        )
        warmup_steps = int(len(train_dataloader) * epochs * warmup_ratio)

        sentence_pairs = [[ex["question"], ex["chunk_text"]] for ex in val_examples]
        labels = [ex["label"] for ex in val_examples]
        epoch_evaluator = CrossEncoderClassificationEvaluator(sentence_pairs, labels, name="finetuned")

        def report_epoch(score, epoch, steps):
            # score is normally a float but handled defensively here in case a given sentence-transformers version passes the full metrics dict instead.
            if isinstance(score, dict):
                score = score.get(f"finetuned_{TARGET_METRIC_SUFFIX}", next(iter(score.values())))

            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        model.fit(
            train_dataloader=train_dataloader,
            evaluator=epoch_evaluator,
            epochs=epochs,
            evaluation_steps=0,  # evaluate at the end of each epoch, not mid-epoch
            warmup_steps=warmup_steps,
            optimizer_params={"lr": lr, "weight_decay": weight_decay},
            callback=report_epoch,
            show_progress_bar=False,
        )

        final_metrics = evaluate_model(model, val_examples, name="finetuned")
        elapsed = time.time() - start_time
        target_value = get_target_metric(final_metrics, "finetuned")

        print(f"Trial {trial.number} done in {elapsed:.1f}s. "
              f"{TARGET_METRIC_SUFFIX}={target_value:.4f}")

        trial.set_user_attr("final_metrics", final_metrics)
        trial.set_user_attr("train_time_seconds", round(elapsed, 1))
        trial.set_user_attr("status", "ok")

        return target_value

    except optuna.TrialPruned:
        elapsed = time.time() - start_time
        trial.set_user_attr("train_time_seconds", round(elapsed, 1))
        trial.set_user_attr("status", "pruned")
        print(f"Trial {trial.number} pruned after {elapsed:.1f}s.")
        raise

    except Exception as e:
        elapsed = time.time() - start_time
        trial.set_user_attr("train_time_seconds", round(elapsed, 1))
        trial.set_user_attr("status", "failed")
        trial.set_user_attr("error", str(e))
        print(f"[FAILED] Trial {trial.number} failed after {elapsed:.1f}s: {e}")
        raise

    finally:
        if model is not None:
            del model
        free_gpu_memory()


# ---------------------------------------------------------------------------
# Results export

def save_study_results(study: optuna.Study, baseline_metrics: dict, output_path: str):
    trials_out = []
    for t in study.trials:
        trials_out.append({
            "trial": t.number,
            "params": t.params,
            "state": t.state.name,  # COMPLETE / PRUNED / FAIL
            "value": t.value,       # target metric, if the trial completed
            "final_metrics": t.user_attrs.get("final_metrics"),
            "train_time_seconds": t.user_attrs.get("train_time_seconds"),
            "error": t.user_attrs.get("error"),
        })

    best_trial_info = None
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        best = study.best_trial
        best_trial_info = {
            "trial": best.number,
            "params": best.params,
            "value": best.value,
            "final_metrics": best.user_attrs.get("final_metrics"),
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline_metrics": baseline_metrics,
                "data_fraction_used": DATA_FRACTION,
                "search_strategy": "optuna (TPE sampler + MedianPruner)",
                "n_trials_requested": N_TRIALS,
                "best_trial": best_trial_info,
                "trials": trials_out,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


class SaveResultsCallback:
    """Optuna study.optimize callback - saves results to disk after every trial."""

    def __init__(self, baseline_metrics: dict, output_path: str):
        self.baseline_metrics = baseline_metrics
        self.output_path = output_path

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        save_study_results(study, self.baseline_metrics, self.output_path)


# ---------------------------------------------------------------------------
# Main

def run_search(n_trials: int = N_TRIALS):
    global train_examples, val_examples

    random.seed(RANDOM_SEED)

    if not os.path.exists(DATA_PATH):
        print(f"file missing at: {DATA_PATH}")
        return

    print("Loading training data...")
    all_examples = load_training_data(DATA_PATH)
    print(f"Loaded {len(all_examples)} pairs (question, chunk).")

    # Reduce to a subset for the search, simple row-level slicing
    random.shuffle(all_examples)
    subset_size = int(len(all_examples) * DATA_FRACTION)
    subset_examples = all_examples[:subset_size]
    print(f"Using {len(subset_examples)}/{len(all_examples)} examples ({DATA_FRACTION:.0%}) for the search.")

    train_examples, val_examples = split_train_val(subset_examples, VAL_SPLIT_RATIO, RANDOM_SEED)
    print(f"SPLIT: {len(train_examples)} train / {len(val_examples)} val")

    # Baseline (no fine-tuning) evaluated once - doesn't depend on
    # hyperparameters, so no need to repeat it per trial.
    print(f"\nEvaluating base model ({BASE_MODEL}) before any fine-tuning...")
    baseline_model = CrossEncoder(BASE_MODEL, num_labels=1)
    baseline_metrics = evaluate_model(baseline_model, val_examples, name="baseline")
    print(f"Baseline metrics: {baseline_metrics}")
    del baseline_model
    free_gpu_memory()

    study = optuna.create_study(
        direction="maximize",  # average_precision: higher is better
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )

    save_callback = SaveResultsCallback(baseline_metrics, OUTPUT_PATH)

    study.optimize(
        objective,
        n_trials=n_trials,
        catch=(RuntimeError,),  # OOM - log as failed, keep going
        callbacks=[save_callback],
    )

    print(f"\nSearch complete. Results saved to: {OUTPUT_PATH}")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        print(f"Best trial: #{study.best_trial.number}, params={study.best_trial.params}, "
              f"{TARGET_METRIC_SUFFIX}={study.best_value:.4f}")
    else:
        print("No trials completed successfully.")


if __name__ == "__main__":
    run_search()