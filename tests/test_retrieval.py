"""
test_retrieval.py - Unit tests for FAISS retrieval behavior.

Tests that the FAISS index properly stores documents and that the
top_k behavior returns the expected number of relevant chunks.
"""

import pytest
from langchain_core.documents import Document

from src.vector_store import build_vector_store
from src.retriever import get_retriever
from src.config import TOP_K_RESULTS


@pytest.fixture
def mock_documents():
    """Provides a small set of mock documents for testing retrieval."""
    return [
        "Patient: What are the symptoms of flu?\nDoctor: Symptoms include fever, chills, muscle aches, cough, congestion, runny nose, headaches, and fatigue.",
        "Patient: Tell me about hypertension.\nDoctor: Hypertension is a condition where the blood pressure in the arteries is persistently elevated.",
        "Patient: How to treat a minor burn?\nDoctor: Treat a minor burn by cooling it with running water, then applying a soothing lotion or aloe vera.",
        "Patient: What causes migraines?\nDoctor: Migraines can be triggered by stress, certain foods, sleep changes, and hormonal fluctuations.",
    ]


def test_build_vector_store(mock_documents):
    """Test that FAISS vector store can be built from documents."""
    vector_store = build_vector_store(mock_documents, save=False)
    assert vector_store is not None
    # Assuming chunk size is large enough to not split these short documents
    # and FAISS successfully stored them
    # Ensure it's queryable
    results = vector_store.similarity_search("flu symptoms", k=1)
    assert len(results) == 1
    assert "fever, chills" in results[0].page_content.lower()


def test_retriever_top_k_behavior(mock_documents):
    """Test that the configured retriever respects top_k settings."""
    vector_store = build_vector_store(mock_documents, save=False)
    retriever = get_retriever(vector_store)
    
    # Temporarily modify top_k in retriever
    retriever.search_kwargs["k"] = 2
    
    results = retriever.invoke("migraines")
    
    # Should return at most 2 results based on our override
    assert len(results) <= 2
    # The most relevant should be the migraine doc
    assert "migraines" in results[0].page_content.lower()
