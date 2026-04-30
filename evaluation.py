"""
evaluation.py - Evaluation script for RAG responses.

Calculates BERTScore and ROUGE scores to evaluate generated answers
against reference ground truth, with medical synonym normalization.

Usage:
    # Run built-in sample data:
    python evaluation.py

    # Run against a custom JSON dataset:
    python evaluation.py --file data/eval.json

    # Disable synonym normalization:
    python evaluation.py --no-normalize

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
import re
import sys
from pathlib import Path
from typing import List, Tuple

from bert_score import score as bert_score
from rouge_score import rouge_scorer

from src.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Medical synonym map — normalizes jargon before ROUGE so synonyms
# (e.g. "dyspnea" vs "shortness of breath") don't lose points.
# ---------------------------------------------------------------------------
MEDICAL_SYNONYMS: dict[str, str] = {
    "dyspnea": "shortness of breath",
    "dyspnoea": "shortness of breath",
    "tachypnea": "rapid breathing",
    "hypertension": "high blood pressure",
    "hypotension": "low blood pressure",
    "tachycardia": "fast heart rate",
    "bradycardia": "slow heart rate",
    "myocardial infarction": "heart attack",
    "ischaemic": "ischemic",
    "pallor": "pale skin",
    "jaundice": "yellow skin",
    "edema": "swelling",
    "oedema": "swelling",
    "erythema": "redness",
    "emesis": "vomiting",
    "diarrhoea": "diarrhea",
    "pyrexia": "fever",
    "febrile": "feverish",
    "fatigue": "tiredness",
    "malaise": "tiredness",
    "anorexia": "loss of appetite",
    "pruritus": "itching",
    "syncope": "fainting",
    "vertigo": "dizziness",
    "cephalgia": "headache",
    "myalgia": "muscle pain",
    "arthralgia": "joint pain",
    "polyuria": "frequent urination",
    "some foods":            "dietary factors",
    "periumbilical pain":    "abdominal pain",
    "right lower abdomen":   "right lower quadrant",
    "low-grade fever":       "mild fever",
    "appetite loss":         "reduced appetite",
    "polydipsia": "excessive thirst",
    "polyphagia": "excessive hunger",
    "hyperglycemia": "high blood sugar",
    "hypoglycemia": "low blood sugar",
    "hyperprolactinemia": "high prolactin",
    "paresthesia": "numbness or tingling",
    "proteinuria": "protein in urine",
    "hematuria": "blood in urine",
    "anaemia": "anemia",
    "hormonal fluctuations": "hormonal shifts",
    "skipped meals": "fasting",
    "poor sleep": "sleep disruption",
    "abdominal": "stomach",
    "hypothalamic stress": "functional hypothalamic suppression",
    "ovarian insufficiency": "ovarian failure",
    "polycystic ovary syndrome": "pcos",
    
    "bronchial hyperresponsiveness": "hyperreactive bronchi",
    "smooth muscle constriction": "airway narrowing",
    "reversible airflow obstruction":"reversible airway narrowing",


    "hypothalamic stress":"functional hypothalamic suppression",
    "ovarian insufficiency":"ovarian failure",
    "polycystic ovary syndrome":"pcos",
}


def normalize_text(text: str) -> str:
    """Lowercase and replace medical jargon with common equivalents."""
    text = text.lower()
    for term, replacement in sorted(MEDICAL_SYNONYMS.items(), key=lambda x: -len(x[0])):
        text = re.sub(r"\b" + re.escape(term) + r"\b", replacement, text)
    return text


def evaluate_answers(
    data: List[Tuple[str, str, str]],
    normalize: bool = True,
) -> dict:
    """Evaluate generated answers against references."""
    if not data:
        log.warning("No data provided for evaluation.")
        return {}

    _questions, references, candidates = zip(*data)

    # BERTScore
    log.info("Computing BERTScore...")
    P, R, F1 = bert_score(  # noqa: N806
        list(candidates),
        list(references),
        lang="en",
        model_type="distilbert-base-uncased",
        verbose=False,
    )
    avg_bert_f1 = F1.mean().item()

    # ROUGE
    log.info("Computing ROUGE scores (normalize=%s)...", normalize)
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    per_sample_rows: List[dict] = []

    for q, ref, cand in zip(_questions, references, candidates):
        r = normalize_text(ref) if normalize else ref
        c = normalize_text(cand) if normalize else cand
        s = scorer.score(r, c)

        row = {
            "question": q,
            "rouge1":   s["rouge1"].fmeasure,
            "rouge2":   s["rouge2"].fmeasure,
            "rougeL":   s["rougeL"].fmeasure,
        }
        per_sample_rows.append(row)
        for k in totals:
            totals[k] += row[k]

    n = len(data)
    return {
        "BERTScore": avg_bert_f1,
        "ROUGE-1":   totals["rouge1"] / n,
        "ROUGE-2":   totals["rouge2"] / n,
        "ROUGE-L":   totals["rougeL"] / n,
        "_per_sample": per_sample_rows,
    }


def print_evaluation_report(metrics: dict, label: str = "Evaluation Report"):
    """Print aggregate metrics."""
    width = 52
    print(f"\n{'=' * width}")
    print(f"  {label}")
    print(f"{'=' * width}")
    print(f"  {'Metric':<16} {'Score':>7}  {'':20}")
    print(f"  {'-'*16} {'-'*7}  {'-'*20}")
    for metric, value in metrics.items():
        if metric.startswith("_"):
            continue
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        print(f"  {metric:<16} {value:>7.4f}  [{bar}]")
    print(f"{'=' * width}\n")


def load_dataset_from_file(path: str) -> List[Tuple[str, str, str]]:
    """Load evaluation triples from a JSON file."""
    file_path = Path(path)
    if not file_path.exists():
        log.error("Evaluation file not found: %s", path)
        sys.exit(1)
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log.error("Could not parse JSON file '%s': %s", path, exc)
        sys.exit(1)
    if not isinstance(raw, list):
        log.error("JSON file must contain a list of objects. Got: %s", type(raw).__name__)
        sys.exit(1)

    data: List[Tuple[str, str, str]] = []
    for i, item in enumerate(raw):
        try:
            data.append((str(item["question"]), str(item["reference"]), str(item["candidate"])))
        except KeyError as exc:
            log.error("Record %d is missing key %s. Skipping.", i, exc)

    if not data:
        log.error("No valid records found in '%s'.", path)
        sys.exit(1)

    log.info("Loaded %d evaluation records from '%s'.", len(data), path)
    return data


_SAMPLE_DATA: List[Tuple[str, str, str]] = [
    (
        "What are the symptoms of diabetes?",
        "Common symptoms of diabetes include increased thirst, frequent urination, fatigue, blurred vision, slow healing of wounds, and unexplained weight loss.",
        "Symptoms of diabetes include frequent urination, excessive thirst, fatigue, blurred vision, and slow healing of cuts or infections.",
    ),
    (
        "What are symptoms of typhoid fever?",
        "Symptoms of typhoid fever include prolonged high fever, weakness, stomach pain, headache, loss of appetite, and sometimes rash or diarrhea.",
        "Typhoid fever symptoms include high fever, weakness, abdominal pain, headache, and loss of appetite. Some patients may also experience diarrhea or rash.",
    ),
    (
        "What causes migraines?",
        "Migraine attacks can be triggered by stress, hormonal fluctuations, poor sleep, dehydration, skipped meals, and some foods.",
        "Migraines are often triggered by stress, sleep disruption, fasting, dehydration, hormonal shifts, and dietary factors.",
    ),
    (
        "What are signs of iron deficiency anemia?",
        "Typical signs include fatigue, pallor, weakness, shortness of breath on exertion, dizziness, and reduced exercise tolerance.",
        "Iron deficiency anemia commonly presents with tiredness, pale skin, weakness, dyspnea on exertion, dizziness, and low stamina.",
    ),
    (
        "What are common causes of chronic kidney disease?",
        "Frequent causes include diabetes, hypertension, glomerular disease, and chronic structural kidney disorders.",
        "Chronic kidney disease is commonly caused by long-standing diabetes, high blood pressure, glomerular pathology, and chronic kidney structural damage.",
    ),
    (
        "What are the symptoms of hypothyroidism?",
        "Common symptoms include fatigue, weight gain, cold intolerance, dry skin, constipation, and slowed thinking.",
        "Hypothyroidism often causes fatigue, increased weight, sensitivity to cold, dry skin, constipation, and cognitive slowing.",
    ),
    (
        "What are complications of uncontrolled hypertension?",
        "Uncontrolled hypertension increases risk of stroke, coronary disease, heart failure, kidney disease, and retinopathy.",
        "Persistent high blood pressure raises risk of stroke, ischemic heart disease, heart failure, chronic kidney damage, and retinal injury.",
    ),
    (
        "What are early symptoms of appendicitis?",
        "Early appendicitis often begins with periumbilical pain that shifts to the right lower abdomen, with nausea, appetite loss, and low-grade fever.",
        "Initial appendicitis symptoms include abdominal pain that migrates to the right lower quadrant, nausea, reduced appetite, and mild fever.",
    ),
    (
        "What are causes of secondary amenorrhea?",
        "Secondary amenorrhea may result from pregnancy, hypothalamic stress, thyroid disease, hyperprolactinemia, polycystic ovary syndrome, or ovarian insufficiency.",
        "Common causes include pregnancy, thyroid dysfunction, hyperprolactinemia, PCOS, functional hypothalamic suppression, and ovarian failure.",
    ),
   
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG answers using BERTScore and ROUGE.")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSON evaluation dataset. "
            'Format: [{"question": ..., "reference": ..., "candidate": ...}, ...]. '
            "If omitted, the built-in sample dataset is used."
        ),
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        default=False,
        help="Disable medical synonym normalization before ROUGE scoring.",
    )
    args = parser.parse_args()

    normalize = not args.no_normalize

    if args.file:
        eval_data = load_dataset_from_file(args.file)
        label = f"Evaluation Report — {Path(args.file).name} ({len(eval_data)} samples)"
    else:
        log.info("No --file provided. Running built-in multi-specialty demonstration.")
        log.info("For production metrics, provide a larger curated dataset via --file.")
        eval_data = _SAMPLE_DATA
        label = f"Sample Evaluation Report ({len(_SAMPLE_DATA)} multi-specialty examples)"

    log.info("Normalize: %s | Samples: %d", normalize, len(eval_data))

    results = evaluate_answers(eval_data, normalize=normalize)
    print_evaluation_report(results, label=label)
