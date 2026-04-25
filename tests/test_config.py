"""
tests/test_config.py – Sanity checks for environment configuration.
"""

import os
import pytest
from src.config import (
    OLLAMA_MODEL,
    EMBEDDING_MODEL,
    TOP_K_RESULTS,
    VECTOR_STORE_PATH,
    DATASET_NAME,
)


def test_ollama_model_is_set():
    assert isinstance(OLLAMA_MODEL, str) and len(OLLAMA_MODEL) > 0


def test_embedding_model_is_set():
    assert isinstance(EMBEDDING_MODEL, str) and len(EMBEDDING_MODEL) > 0


def test_top_k_is_positive_int():
    assert isinstance(TOP_K_RESULTS, int) and TOP_K_RESULTS > 0


def test_vector_store_path_is_path_object():
    from pathlib import Path
    assert isinstance(VECTOR_STORE_PATH, Path)


def test_dataset_name_is_set():
    assert isinstance(DATASET_NAME, str) and len(DATASET_NAME) > 0
