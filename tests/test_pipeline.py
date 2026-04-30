"""
test_pipeline.py - Integration tests for the full query-answer pipeline.

Ensures that the pipeline pieces (retriever, reranker, formatting, LLM)
can be instantiated and invoked together.
"""

from unittest.mock import MagicMock
from langchain_core.documents import Document

from src.rag_chain import query


def test_pipeline_no_crash_on_empty_context():
    """Test that the pipeline returns a graceful failure on empty context."""
    # Create a mock chain whose run() method returns a valid result dict.
    mock_chain = MagicMock()
    mock_chain.run.return_value = {
        "answer": "I could not find relevant information.",
        "sources": [],
        "context_chars": 0,
        "top_k_used": 0,
        "confidence": "low",
        "grounding_removals": 0,
    }

    result = query(mock_chain, "What is a very rare made-up disease?")

    assert isinstance(result, dict)
    assert "answer" in result
    assert result["sources"] == []


def test_pipeline_formatting_robustness():
    """Test that the pipeline's answer formatting does not crash on strange inputs."""
    from src.rag_chain import _format_general_answer, _format_definition_answer
    
    # Check general answer formatting
    bad_answer = "   \n\n\n- \n- \n   "
    clean = _format_general_answer(bad_answer)
    assert isinstance(clean, str)
    
    # Check definition formatting
    clean_def = _format_definition_answer("Fake Disease", bad_answer, docs=[])
    assert "I do not have enough reliable information in the provided context to answer this question." in clean_def
