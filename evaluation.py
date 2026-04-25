"""
evaluation.py - Evaluation script for RAG responses.

Calculates BERTScore and ROUGE scores to evaluate generated answers
against reference ground truth.

Usage:
    # Run built-in sample data (3 examples for quick sanity check):
    python evaluation.py

    # Run against a custom JSON dataset file:
    python evaluation.py --file data/eval.json

Expected JSON format for --file:
    [
        {
            "question": "What are symptoms of diabetes?",
            "reference": "Common symptoms include ...",
            "candidate": "Symptoms of diabetes include ..."
        },
        ...
    ]

Requirements:
    pip install bert-score rouge-score
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

from bert_score import score as bert_score
from rouge_score import rouge_scorer

from src.logger import get_logger

log = get_logger(__name__)


def evaluate_answers(data: List[Tuple[str, str, str]]) -> dict:
    """
    Evaluates generated answers against reference answers.

    Args:
        data: A list of tuples (question, reference_answer, generated_answer).

    Returns:
        A dictionary containing averaged evaluation metrics.
    """
    if not data:
        log.warning("No data provided for evaluation.")
        return {}

    questions, references, candidates = zip(*data)

    log.info("Computing BERTScore...")
    # distilbert is used for speed during development.
    # Switch to model_type="roberta-large" for final/publication metrics.
    P, R, F1 = bert_score(
        candidates, references,
        lang="en",
        model_type="distilbert-base-uncased",
        verbose=False,
    )

    avg_bert_f1 = F1.mean().item()

    log.info("Computing ROUGE scores...")
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True,
    )

    total_rouge1, total_rouge2, total_rougeL = 0.0, 0.0, 0.0

    for ref, cand in zip(references, candidates):
        scores = scorer.score(ref, cand)
        total_rouge1 += scores['rouge1'].fmeasure
        total_rouge2 += scores['rouge2'].fmeasure
        total_rougeL += scores['rougeL'].fmeasure

    num_samples = len(data)

    metrics = {
        "BERTScore": avg_bert_f1,
        "ROUGE-1": total_rouge1 / num_samples,
        "ROUGE-2": total_rouge2 / num_samples,
        "ROUGE-L": total_rougeL / num_samples,
    }

    return metrics


def print_evaluation_report(metrics: dict, label: str = "Evaluation Report"):
    """Prints a cleanly formatted evaluation report with an optional label."""
    print(f"\n--- {label} ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    print("-" * (len(label) + 8) + "\n")


def load_dataset_from_file(path: str) -> List[Tuple[str, str, str]]:
    """
    Load evaluation data from a JSON file.

    The file must be a JSON array of objects with keys:
        "question", "reference", "candidate"

    Args:
        path: Path to the JSON file.

    Returns:
        List of (question, reference, candidate) tuples.

    Raises:
        SystemExit: If the file cannot be read or has the wrong format.
    """
    file_path = Path(path)
    if not file_path.exists():
        log.error("Evaluation file not found: %s", path)
        sys.exit(1)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        log.error("Could not parse JSON file '%s': %s", path, exc)
        sys.exit(1)

    if not isinstance(raw, list):
        log.error("JSON file must contain a list of objects. Got: %s", type(raw).__name__)
        sys.exit(1)

    data: List[Tuple[str, str, str]] = []
    for i, item in enumerate(raw):
        try:
            data.append((
                str(item["question"]),
                str(item["reference"]),
                str(item["candidate"]),
            ))
        except KeyError as exc:
            log.error("Record %d is missing key %s. Skipping.", i, exc)

    if not data:
        log.error("No valid records found in '%s'.", path)
        sys.exit(1)

    log.info("Loaded %d evaluation records from '%s'.", len(data), path)
    return data


# Built-in sample dataset for quick sanity checks.
# Note: only 3 examples — not statistically meaningful.
# Use --file with a larger golden dataset for real evaluation.
_SAMPLE_DATA = [
    (
        "What are the symptoms of diabetes?",
        "Common symptoms of diabetes include increased thirst, frequent urination, "
        "fatigue, blurred vision, slow healing of wounds, and unexplained weight loss.",
        "Symptoms of diabetes include frequent urination, excessive thirst, fatigue, "
        "blurred vision, and slow healing of cuts or infections.",
    ),
    (
        "What are the symptoms of typhoid?",
        "Symptoms of typhoid fever include prolonged high fever, weakness, stomach pain, "
        "headache, loss of appetite, and sometimes rash or diarrhea.",
        "Typhoid symptoms include high fever, weakness, abdominal pain, headache, and "
        "loss of appetite. Some patients may also experience diarrhea or rash.",
    ),
    (
        "What are different types of heart disease?",
        "Heart disease includes conditions such as coronary artery disease, heart attack, "
        "cardiomyopathy, valve disorders, arrhythmias, and atherosclerosis.",
        "Heart disease refers to various conditions affecting the heart and blood vessels. "
        "These include myocardial infarction (heart attack), hypertension, atherosclerosis, "
        "cardiomyopathy, valve diseases such as mitral regurgitation and aortic stenosis, "
        "pulmonary hypertension, arrhythmias, and coronary artery disease.",
    ),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RAG answers using BERTScore and ROUGE.",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSON evaluation dataset. "
            "Format: [{\"question\": ..., \"reference\": ..., \"candidate\": ...}, ...] "
            "If omitted, the built-in 3-sample dataset is used."
        ),
    )
    args = parser.parse_args()

    if args.file:
        eval_data = load_dataset_from_file(args.file)
        label = f"Evaluation Report ({Path(args.file).name}, {len(eval_data)} samples)"
    else:
        log.info("No --file provided. Running built-in 3-sample demonstration.")
        log.info("For meaningful metrics, supply a real dataset with --file.")
        eval_data = _SAMPLE_DATA
        label = "Sample Evaluation Report (3 examples — not statistically significant)"

    log.info("Running evaluation on %d sample(s)...", len(eval_data))
    results = evaluate_answers(eval_data)
    print_evaluation_report(results, label=label)
