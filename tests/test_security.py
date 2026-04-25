"""
tests/test_security.py – Security and validation tests.

Covers the audit findings that were not addressed by existing tests:
    - Input length validation at the API layer
    - Empty query rejection
    - Grounding integrity (bullets no longer unconditionally bypass checks)
    - Context budget enforcement
    - Rate limiter behaviour
    - Config boundary values
"""

import collections
import time

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


# ── 1. Input length validation ────────────────────────────────────────────────

def test_api_rejects_query_over_max_length():
    """POST /query with a query longer than MAX_QUERY_LENGTH must return HTTP 400."""
    from fastapi.testclient import TestClient
    from app.api_server import app
    from src.config import MAX_QUERY_LENGTH

    client = TestClient(app, raise_server_exceptions=False)

    long_query = "A" * (MAX_QUERY_LENGTH + 1)
    response = client.post("/query", json={"question": long_query})

    assert response.status_code == 400
    assert str(MAX_QUERY_LENGTH) in response.json()["detail"]


def test_api_accepts_query_at_max_length():
    """A query exactly at MAX_QUERY_LENGTH characters must not be rejected on length."""
    from fastapi.testclient import TestClient
    from app.api_server import app
    from src.config import MAX_QUERY_LENGTH

    # Patch the chain so we don't need a real vector store for this test.
    mock_result = {
        "answer": "Test answer.",
        "sources": [],
        "confidence": "low",
        "grounding_removals": 0,
    }

    with patch("app.api_server._get_chain") as mock_chain:
        mock_chain.return_value.run = MagicMock(return_value=mock_result)

        # Override query() at the module level used by handle_query
        with patch("app.api_server.query", return_value=mock_result):
            client = TestClient(app, raise_server_exceptions=False)
            exact_query = "B" * MAX_QUERY_LENGTH
            response = client.post("/query", json={"question": exact_query})

    # Should not be rejected for length (may fail for other reasons in test env)
    assert response.status_code != 400 or "too long" not in response.json().get("detail", "")


# ── 2. Empty query rejection ──────────────────────────────────────────────────

def test_api_rejects_empty_query():
    """POST /query with an empty string must return HTTP 400."""
    from fastapi.testclient import TestClient
    from app.api_server import app

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/query", json={"question": ""})

    assert response.status_code == 400


def test_api_rejects_whitespace_only_query():
    """POST /query with only spaces must return HTTP 400."""
    from fastapi.testclient import TestClient
    from app.api_server import app

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/query", json={"question": "   "})

    assert response.status_code == 400


# ── 3. Grounding integrity — bullets no longer blindly pass ──────────────────

def test_hallucinated_bullet_is_removed_by_grounding():
    """
    A bullet point whose content has zero keyword overlap with the context
    must be stripped by _validate_grounding.

    Before the fix, all bullet lines were unconditionally kept.
    """
    from src.rag_chain import _validate_grounding

    docs = [
        Document(
            page_content=(
                "Malaria is caused by Plasmodium parasites transmitted "
                "through the bites of infected female Anopheles mosquitoes."
            )
        )
    ]

    raw_answer = (
        "Malaria is a parasitic infection.\n"
        "- Transmitted by Anopheles mosquitoes\n"        # grounded
        "- Cured by drinking alkaline water daily\n"     # hallucinated bullet
        "- Caused by Plasmodium parasites\n"             # grounded
    )

    clean, removals = _validate_grounding(raw_answer, docs)

    assert "alkaline water" not in clean.lower(), (
        "Hallucinated bullet should have been removed by grounding."
    )
    assert "mosquitoes" in clean.lower(), "Grounded bullet should be preserved."
    assert removals >= 1, "At least one removal should have been recorded."


def test_grounded_bullet_is_preserved():
    """A bullet whose content overlaps with the context must be kept."""
    from src.rag_chain import _validate_grounding

    docs = [
        Document(
            page_content=(
                "Hypertension symptoms include severe headache, blurred vision, "
                "and chest pain."
            )
        )
    ]

    raw_answer = (
        "Hypertension can present with:\n"
        "- Severe headache\n"
        "- Blurred vision\n"
        "- Chest pain\n"
    )

    clean, removals = _validate_grounding(raw_answer, docs)

    assert "headache" in clean.lower()
    assert "blurred vision" in clean.lower()
    assert "chest pain" in clean.lower()


# ── 4. Context budget enforcement ─────────────────────────────────────────────

def test_context_budget_drops_longest_chunk():
    """_enforce_context_budget must drop the longest chunk to fit within budget."""
    from src.rag_chain import _enforce_context_budget

    short_doc = Document(page_content="Short text.")
    long_doc  = Document(page_content="X" * 5000)

    kept, total_chars = _enforce_context_budget([short_doc, long_doc], budget_chars=100)

    assert long_doc not in kept, "The oversized chunk must be dropped."
    assert total_chars <= 100


def test_context_budget_keeps_all_when_within_limit():
    """No chunks must be dropped when total length is within the budget."""
    from src.rag_chain import _enforce_context_budget

    docs = [
        Document(page_content="Fever is common in malaria."),
        Document(page_content="Headache is also a symptom."),
    ]

    kept, total = _enforce_context_budget(docs, budget_chars=5000)

    assert len(kept) == len(docs)


# ── 5. Rate limiter ───────────────────────────────────────────────────────────

def test_rate_limiter_blocks_after_limit():
    """The in-memory rate limiter must block requests that exceed RATE_LIMIT_PER_MIN."""
    from app.api_server import _is_rate_limited
    from src.config import RATE_LIMIT_PER_MIN

    test_ip = "10.0.0.255"  # Use an IP not used by other tests

    # Exhaust the rate limit
    for _ in range(RATE_LIMIT_PER_MIN):
        _is_rate_limited(test_ip)

    # The next request should be blocked
    assert _is_rate_limited(test_ip) is True


def test_rate_limiter_allows_requests_within_limit():
    """Requests below the rate limit must not be blocked."""
    from app.api_server import _is_rate_limited
    from src.config import RATE_LIMIT_PER_MIN

    test_ip = "10.0.1.255"  # Fresh IP

    # The first request must always be allowed
    result = _is_rate_limited(test_ip)
    assert result is False


# ── 6. Config boundary values ─────────────────────────────────────────────────

def test_max_query_length_is_positive():
    """MAX_QUERY_LENGTH must be a positive integer."""
    from src.config import MAX_QUERY_LENGTH
    assert isinstance(MAX_QUERY_LENGTH, int) and MAX_QUERY_LENGTH > 0


def test_rate_limit_per_min_is_positive():
    """RATE_LIMIT_PER_MIN must be a positive integer."""
    from src.config import RATE_LIMIT_PER_MIN
    assert isinstance(RATE_LIMIT_PER_MIN, int) and RATE_LIMIT_PER_MIN > 0


def test_safe_deserialization_is_bool():
    """SAFE_DESERIALIZATION must be a boolean value."""
    from src.config import SAFE_DESERIALIZATION
    assert isinstance(SAFE_DESERIALIZATION, bool)
