"""
test_grounding.py - Unit tests for grounding validation logic.

Ensures that the hallucination-prevention layer correctly identifies
and strips out text not supported by the context, without crashing.
"""

from langchain_core.documents import Document

from src.rag_chain import _validate_grounding, _sentence_is_grounded


def test_sentence_is_grounded():
    """Test n-gram based grounding logic at the sentence level."""
    context = "Typhoid fever is a bacterial infection caused by Salmonella typhi, spread through contaminated food and water."

    # Grounded sentence — shares n-grams and keywords with context.
    grounded = "Typhoid fever is caused by a bacterial infection from Salmonella typhi."
    assert _sentence_is_grounded(grounded, context) is True

    # Truly hallucinated sentence — uses completely different vocabulary;
    # no meaningful word from the sentence appears in the context.
    # Note: sentences sharing topic keywords (e.g. "Typhoid") are intentionally
    # kept by the keyword safety net even at low n-gram overlap.
    hallucinated = "Quantum resonance eliminates carbonic flux in patients."
    assert _sentence_is_grounded(hallucinated, context) is False

    # Very short sentence (under _GROUNDING_MIN_WORDS) — always kept
    short = "This is it."
    assert _sentence_is_grounded(short, context) is True


def test_validate_grounding():
    """Test the full grounding validator on a sample LLM output."""
    docs = [
        Document(page_content="Migraines can be triggered by stress, lack of sleep, and certain foods.")
    ]
    
    raw_answer = (
        "Migraines can be triggered by stress.\n"
        "They are also caused by drinking too much soda.\n"
        "- lack of sleep\n"
        "- certain foods"
    )
    
    clean_answer, removals = _validate_grounding(raw_answer, docs)
    
    # Should strip the soda line, keep the grounded prose, and keep the bullets
    assert "soda" not in clean_answer.lower()
    assert "stress" in clean_answer.lower()
    assert "lack of sleep" in clean_answer.lower()
    assert removals == 1
