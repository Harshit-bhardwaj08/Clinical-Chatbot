"""
data_loader.py – Downloads and prepares the clinical QA dataset.

This module is responsible for:
  - Fetching the HuggingFace medical dataset (e.g., medical_meadow)
    with automatic retry on transient failures.
  - Loading a secondary ChatDoctor dataset for clinical reasoning.
  - Formatting each record into a readable Patient/Doctor dialogue string.
  - Persisting the raw dataset to disk for inspection.
  - Returning the cleaned documents as a plain Python list ready for embedding.

Usage (standalone):
    python -m src.data_loader

Usage (as a module):
    from src.data_loader import load_clinical_documents
    docs = load_clinical_documents()
"""

import json
import time
from pathlib import Path

from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError

from src.config import (
    DATASET_NAME,
    DATASET_SPLIT,
    DATASET_SUBSET,
    RAW_DATA_PATH,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
)
from src.logger import get_logger

log = get_logger(__name__)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _fetch_dataset_with_retry(name: str, subset: str, split: str):
    """
    Attempt to load a HuggingFace dataset up to MAX_RETRIES times.

    Raises:
        RuntimeError: If all retry attempts are exhausted.
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(
                "Attempt %d/%d – loading '%s' (subset=%s, split=%s)…",
                attempt, MAX_RETRIES, name, subset, split,
            )
            dataset = load_dataset(name, subset, split=split)
            log.info("Dataset fetched successfully on attempt %d.", attempt)
            return dataset

        except DatasetNotFoundError as exc:
            # This is a permanent error – no point retrying.
            log.error(
                "Dataset '%s' (subset='%s') was not found on HuggingFace Hub. ",
                name, subset,
            )
            raise RuntimeError(f"Dataset '{name}' not found.") from exc

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            log.warning("Attempt %d failed: %s", attempt, exc)
            if attempt < MAX_RETRIES:
                log.info("Waiting %s second(s) before retrying…", RETRY_DELAY_SECONDS)
                time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"Failed to load dataset '{name}' after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


def _format_document(row: dict) -> str | None:
    """
    Dynamically extract instruction, input, and output from the row.
    Combines fields to form a Patient/Doctor dialogue string.
    """
    # 1. Detect question
    question = ""
    for q_field in ["input", "instruction", "question", "front"]:
        val = row.get(q_field, "")
        if val and isinstance(val, str) and val.strip():
            question = val.strip()
            break
            
    # 2. Detect answer
    answer = ""
    for a_field in ["answer_icliniq", "output", "answer", "back"]:
        val = row.get(a_field, "")
        if val and isinstance(val, str) and val.strip():
            answer = val.strip()
            break

    if not question or not answer:
        return None

    # 3. Clinical quality validation
    # Reject extremely short answers that lack medical depth
    if len(answer) < 30:
        return None
        
    # Strict validation: no empty text
    if len(question) < 5:
        return None

    # Replace newlines with spaces or keep them? The prompt doesn't strictly say, 
    # but strip whitespace was requested for filters
    question = question.strip()
    answer = answer.strip()

    return f"Patient: {question}\nDoctor: {answer}"


# ── Public API ────────────────────────────────────────────────────────────────


def load_multiple_datasets() -> list[str]:
    """
    Load multiple clinical datasets and return a combined list of formatted documents.
    """
    log.info("=" * 60)
    log.info("Starting multiple dataset load.")
    log.info("=" * 60)

    # 1. Load existing dataset (Medical Meadow)
    meadow_docs = []
    try:
        meadow_dataset = _fetch_dataset_with_retry(DATASET_NAME, DATASET_SUBSET, DATASET_SPLIT)
        for row in meadow_dataset:
            doc = _format_document(row)
            if doc:
                meadow_docs.append(doc)
    except Exception as e:
        log.error("Failed to load Medical Meadow dataset: %s", e)

    # 2. Load ChatDoctor dataset
    chatdoctor_docs = []
    try:
        chatdoctor_dataset = _fetch_dataset_with_retry("lavita/medical-qa-datasets", "chatdoctor-icliniq", "test")
        for row in chatdoctor_dataset:
            doc = _format_document(row)
            if doc:
                chatdoctor_docs.append(doc)
    except Exception as e:
        log.error("Failed to load ChatDoctor dataset: %s", e)

    # 3. Load WikiDoc dataset (general definitions, symptoms)
    wikidoc_docs = []
    try:
        wikidoc_dataset = _fetch_dataset_with_retry("lavita/medical-qa-datasets", "medical_meadow_wikidoc", "train")
        for row in wikidoc_dataset:
            doc = _format_document(row)
            if doc:
                wikidoc_docs.append(doc)
    except Exception as e:
        log.error("Failed to load WikiDoc dataset: %s", e)
        
    # 4. Load MedQA dataset (general textbook knowledge)
    medqa_docs = []
    try:
        medqa_dataset = _fetch_dataset_with_retry("lavita/medical-qa-datasets", "medical_meadow_medqa", "train")
        for row in medqa_dataset:
            doc = _format_document(row)
            if doc:
                medqa_docs.append(doc)
    except Exception as e:
        log.error("Failed to load MedQA dataset: %s", e)

    # 5. Rebalance dataset to prioritize general clinical knowledge
    # Target: ~50-60% general knowledge vs 40-50% case-based/specialized
    wikidoc_docs = wikidoc_docs[:3000]
    medqa_docs = medqa_docs[:1000]

    # Downsample specialized/case-based datasets to prevent dominance
    chatdoctor_docs = chatdoctor_docs[:2000]
    meadow_docs = meadow_docs[:2000]

    # Step 4 - Merge Datasets
    combined_docs = meadow_docs + chatdoctor_docs + wikidoc_docs + medqa_docs
    
    if not combined_docs:
        raise RuntimeError("All datasets failed to load or yielded 0 valid documents.")

    # Step 5 - Quality Filter (Remove duplicates)
    # Use set to remove duplicates, preserve order by using dict.fromkeys in python 3.7+
    unique_docs = list(dict.fromkeys(combined_docs))

    # Step 6 - Logging (Mandatory)
    log.info("Loaded Meadow docs: %d", len(meadow_docs))
    log.info("Loaded ChatDoctor docs: %d", len(chatdoctor_docs))
    log.info("Loaded WikiDoc docs: %d", len(wikidoc_docs))
    log.info("Loaded MedQA docs: %d", len(medqa_docs))
    log.info("Final merged docs: %d", len(unique_docs))

    if unique_docs:
        log.info("--- Sample document (first record) ---")
        log.info("\n%s", unique_docs[0])
        log.info("--------------------------------------")

    return unique_docs


def load_clinical_documents() -> list[str]:
    """Alias for backwards compatibility with main.py"""
    return load_multiple_datasets()


# ── Standalone entry point ────────────────────────────────────────────────────


def main() -> None:
    """Run the data loader on its own and print one formatted document."""
    print("\n[ Clinical RAG – Data Loader ]\n")

    try:
        docs = load_multiple_datasets()
    except RuntimeError as err:
        print(f"\n[ERROR] Could not load datasets:\n  {err}")
        return

    print(f"\n[OK] Loaded {len(docs)} documents successfully.\n")

    if docs:
        print("--- Sample Formatted Document ---------------------------------")
        print(docs[0])
        print("---------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
