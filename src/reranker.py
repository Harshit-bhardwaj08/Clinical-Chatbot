"""Second-stage semantic reranking for the clinical RAG pipeline.

This module keeps the same public API (`rerank_docs`) while removing
query-template and symptom-list heuristics. Ranking is now data-driven:
1) semantic similarity to the query,
2) generic medical relevance density in each chunk,
3) coverage of multiple query facets.
"""

from __future__ import annotations

import math
import re
from typing import NamedTuple

from src.config import (
    RERANK_FACET_SIM_THRESHOLD,
    RERANK_MIN_CHUNKS,
    RERANK_MIN_SCORE,
    RERANK_TOP_N,
    RERANK_WEIGHT_FACET_COVERAGE,
    RERANK_WEIGHT_MEDICAL_RELEVANCE,
    RERANK_WEIGHT_SEMANTIC,
)
from src.logger import get_logger

log = get_logger(__name__)

_EMBED_MODEL = None
_EMBED_CACHE: dict[str, list[float]] = {}

# Generic stop words for token cleanup (not disease-specific).
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
        "of", "for", "with", "by", "from", "as", "is", "was", "are", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall", "can",
        "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
        "that", "this", "these", "those", "it", "its", "he", "she", "they",
        "we", "you", "i", "me", "my", "your", "his", "her", "our", "their",
        "what", "which", "who", "whom", "how", "when", "where", "why",
        "about", "above", "after", "before", "between", "during", "through",
        "without", "within", "along", "following", "across", "behind",
        "beyond", "plus", "except", "up", "out", "around", "down", "off",
        "over", "then", "once", "please", "hello", "hi", "thanks",
    }
)
_MIN_TOKEN_LEN = 3

# Generic phrase cues used to split multi-facet symptom/cause queries.
_FACET_SPLIT_PATTERN = re.compile(r",|;|\band\b|\bwith\b|\bplus\b", re.IGNORECASE)

# Generic medical morphology and units (dataset-agnostic, no disease names).
_MEDICAL_AFFIX_PATTERN = re.compile(
    r"(itis|emia|osis|opathy|algia|genic|scopy|ectomy|plasty|oma|uria|pnea|rrhea|pathy)$",
    re.IGNORECASE,
)
_CLINICAL_UNIT_PATTERN = re.compile(r"\b(\d+(\.\d+)?\s*(mg|mcg|ml|mmhg|bpm|kg|cm|mmol|iu))\b", re.IGNORECASE)


class ScoredDoc(NamedTuple):
    """A retrieved document paired with its reranking score."""

    doc: object
    score: float
    semantic_score: float
    medical_relevance: float
    facet_coverage: float


def _get_embedding_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from src.embedder import get_embedding_model

        _EMBED_MODEL = get_embedding_model()
    return _EMBED_MODEL


def _embed(text: str) -> list[float]:
    key = (text or "").strip().lower()
    if key in _EMBED_CACHE:
        return _EMBED_CACHE[key]
    emb = _get_embedding_model().embed_query(text or "")
    _EMBED_CACHE[key] = emb
    return emb


def _cosine(vec1: list[float], vec2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def extract_keywords(query: str) -> list[str]:
    """Extract content-bearing tokens from query text."""
    raw_tokens = re.split(r"[^a-zA-Z0-9\-]+", (query or "").lower())
    seen: set[str] = set()
    keywords: list[str] = []
    for tok in raw_tokens:
        tok = tok.strip("-")
        if len(tok) >= _MIN_TOKEN_LEN and tok not in _STOP_WORDS and tok not in seen:
            seen.add(tok)
            keywords.append(tok)
    return keywords


def _extract_query_facets(query: str) -> list[str]:
    """Extract independent query facets for multi-symptom/multi-clause coverage."""
    clauses = [c.strip() for c in _FACET_SPLIT_PATTERN.split(query or "") if c.strip()]
    facets: list[str] = []
    for clause in clauses:
        kws = extract_keywords(clause)
        if kws:
            facets.append(" ".join(kws[:8]))
    # Deduplicate preserving order.
    return list(dict.fromkeys(facets))


def _medical_relevance_density(text: str) -> float:
    """Score how clinically substantive a chunk is without disease-specific dictionaries."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", (text or "").lower())
    if not tokens:
        return 0.0

    affix_hits = sum(1 for t in tokens if _MEDICAL_AFFIX_PATTERN.search(t))
    unit_hits = len(_CLINICAL_UNIT_PATTERN.findall(text or ""))
    long_term_hits = sum(1 for t in tokens if len(t) >= 9)
    ratio = (affix_hits + unit_hits + long_term_hits * 0.2) / max(len(tokens), 1)
    return max(0.0, min(1.0, ratio * 6.0))


def _facet_coverage_score(doc_text: str, facet_embeddings: list[list[float]]) -> float:
    if not facet_embeddings:
        return 0.0
    doc_emb = _embed(doc_text)
    covered = 0
    for facet_emb in facet_embeddings:
        if _cosine(doc_emb, facet_emb) >= RERANK_FACET_SIM_THRESHOLD:
            covered += 1
    return covered / len(facet_embeddings)


def score_doc(doc, query: str, query_embedding: list[float] | None = None, facet_embeddings: list[list[float]] | None = None) -> ScoredDoc:
    """Compute composite semantic rerank score for one retrieved chunk."""
    text = getattr(doc, "page_content", "") or ""
    if query_embedding is None:
        query_embedding = _embed(query)
    if facet_embeddings is None:
        facets = _extract_query_facets(query)
        facet_embeddings = [_embed(f) for f in facets if f]

    doc_embedding = _embed(text)
    semantic_score = max(0.0, _cosine(query_embedding, doc_embedding))
    medical_relevance = _medical_relevance_density(text)
    facet_coverage = _facet_coverage_score(text, facet_embeddings)

    final_score = (
        RERANK_WEIGHT_SEMANTIC * semantic_score
        + RERANK_WEIGHT_MEDICAL_RELEVANCE * medical_relevance
        + RERANK_WEIGHT_FACET_COVERAGE * facet_coverage
    )
    final_score = max(0.0, min(1.0, final_score))

    return ScoredDoc(
        doc=doc,
        score=final_score,
        semantic_score=semantic_score,
        medical_relevance=medical_relevance,
        facet_coverage=facet_coverage,
    )


def rerank_docs(docs: list, query: str) -> list:
    """Filter and rank retrieved documents by semantic and clinical relevance."""
    if not docs:
        log.warning("Reranker received empty document list; skipping.")
        return docs

    query_embedding = _embed(query)
    query_facets = _extract_query_facets(query)
    facet_embeddings = [_embed(f) for f in query_facets]

    scored = [
        score_doc(
            doc,
            query=query,
            query_embedding=query_embedding,
            facet_embeddings=facet_embeddings,
        )
        for doc in docs
    ]
    scored.sort(key=lambda sd: sd.score, reverse=True)

    filtered = [sd for sd in scored if sd.score >= RERANK_MIN_SCORE]
    if not filtered:
        log.warning(
            "All %d chunks were below score threshold %.2f; applying safety net.",
            len(scored),
            RERANK_MIN_SCORE,
        )
        filtered = scored[: max(RERANK_MIN_CHUNKS, 1)]

    min_docs = min(max(RERANK_MIN_CHUNKS, 1), len(scored))
    if len(filtered) < min_docs:
        filtered = scored[:min_docs]

    if RERANK_TOP_N > 0:
        filtered = filtered[:RERANK_TOP_N]

    log.info(
        "Reranker: %d -> %d chunks (threshold=%.2f, top_n=%d).",
        len(docs),
        len(filtered),
        RERANK_MIN_SCORE,
        RERANK_TOP_N,
    )
    return [sd.doc for sd in filtered]

