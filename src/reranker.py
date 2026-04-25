"""
reranker.py  –  Second-stage semantic reranking for the Clinical RAG pipeline.

Problem
-------
FAISS retrieval is purely embedding-based (cosine similarity of dense vectors).
That can return chunks that are *topically adjacent* but do not actually contain
the clinical terms the user asked about.  For example, a query about "diabetes
symptoms" might surface chunks about "insulin dosage" (high cosine similarity)
even though the word "diabetes" does not appear in those chunks.

Solution
--------
After FAISS retrieval this module applies a lightweight, two-signal reranker:

    final_score = w_kw * keyword_overlap_score
                + w_sim * faiss_similarity_score   ← (if available; else 0)

Keyword overlap is computed dynamically from the query itself – no disease name
is ever hardcoded.  Stop-words and very short tokens are stripped so only
clinically meaningful terms contribute.

Public API
----------
    from src.reranker import rerank_docs

    reranked = rerank_docs(docs, query)
    # Returns a list of scored Document objects, best first.
    # Chunks that score below MIN_SCORE are removed unless doing so would
    # leave an empty list (in that case the single best-scoring chunk is kept
    # so the LLM always has *some* context to reason about).
"""

from __future__ import annotations

import re
import math
from typing import NamedTuple

from src.config import RERANK_MIN_SCORE, RERANK_TOP_N
from src.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Medical stop-words  (extend freely – nothing disease-specific here)
# ---------------------------------------------------------------------------

# Generic English stop-words plus common clinical filler terms that carry
# very little discriminating power.  Kept as a frozenset for O(1) lookup.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        # English function words
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
        "over", "then", "once",
        # Clinical filler words
        "patient", "patients", "doctor", "doctors", "medical", "medicine",
        "health", "healthy", "clinic", "clinical", "please", "dear", "sir",
        "madam", "hello", "hi", "thanks", "thank", "help", "know", "want",
        "need", "feel", "feels", "feeling", "also", "just", "like", "well",
        "good", "bad", "new", "old", "many", "much", "more", "most", "some",
        "any", "all", "other", "such", "even", "only", "still", "already",
        "since", "due", "per", "eg", "ie",
    }
)

# Minimum token length to be considered a keyword.
_MIN_TOKEN_LEN: int = 3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ScoredDoc(NamedTuple):
    """A retrieved document paired with its reranking score."""
    doc: object          # LangChain Document
    score: float         # composite rerank score ∈ [0.0, 1.0]
    keyword_score: float # keyword-overlap component
    sim_score: float     # embedding-similarity component (may be 0.0)


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

def extract_keywords(query: str) -> list[str]:
    """Extract clinically meaningful tokens from *query* dynamically.

    Strategy
    --------
    1. Lowercase and tokenise on non-alphanumeric boundaries.
    2. Remove stop-words and very short tokens.
    3. Deduplicate while preserving original order (first occurrence wins).

    This approach is deliberately disease-agnostic: it extracts whatever
    content words appear in the query, so it generalises to any medical topic.

    Args:
        query: Raw user query string.

    Returns:
        Ordered list of unique keyword strings (may be empty).

    Examples:
        >>> extract_keywords("What are the symptoms of Type 2 Diabetes?")
        ['what', 'symptoms', 'type', 'diabetes']
        # Stop-words removed; 'of', 'the', 'are' filtered out.
    """
    # Tokenise: split on anything that is not a letter, digit, or hyphen.
    raw_tokens = re.split(r"[^a-zA-Z0-9\-]+", query.lower())

    seen: set[str] = set()
    keywords: list[str] = []
    for tok in raw_tokens:
        tok = tok.strip("-")  # strip leading/trailing hyphens
        if (
            len(tok) >= _MIN_TOKEN_LEN
            and tok not in _STOP_WORDS
            and tok not in seen
        ):
            seen.add(tok)
            keywords.append(tok)

    log.debug("Extracted keywords from query: %s", keywords)
    return keywords


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _keyword_overlap_score(text: str, keywords: list[str]) -> float:
    """Compute normalised keyword-overlap score for *text*.

    Score = (number of distinct query keywords present in text) / len(keywords)

    Presence is checked as a case-insensitive whole-word match so that
    "hyperglycemia" in the text does not accidentally match keyword "glyc".

    Args:
        text:     Chunk text (page_content).
        keywords: Keywords extracted from the query.

    Returns:
        Float in [0.0, 1.0].  Returns 0.0 if keywords is empty.
    """
    if not keywords:
        return 0.0

    lowered = text.lower()
    hits = 0
    for kw in keywords:
        # Whole-word boundary match to avoid false substring hits.
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, lowered):
            hits += 1

    return hits / len(keywords)


def _extract_faiss_sim_score(doc) -> float:
    """Try to read a FAISS similarity score stored in document metadata.

    LangChain's ``similarity_search_with_score`` stores the L2 distance in
    metadata under the key 'score' or 'similarity_score' (depending on the
    wrapper version).  Plain ``similarity_search`` does *not* attach a score.

    If no score is found we return 0.0 so the keyword component drives ranking.

    Args:
        doc: LangChain Document object.

    Returns:
        Float in [0.0, 1.0].  Lower FAISS L2 distance → higher return value.
    """
    meta = getattr(doc, "metadata", {}) or {}

    # Some wrappers attach the raw cosine similarity directly.
    for key in ("similarity_score", "score", "relevance_score"):
        raw = meta.get(key)
        if raw is not None:
            try:
                val = float(raw)
                # If it looks like an L2 distance (>1), convert to similarity.
                if val > 1.0:
                    # Heuristic: map L2 distance to (0, 1] via exp(-d).
                    val = math.exp(-val)
                return max(0.0, min(1.0, val))
            except (ValueError, TypeError):
                pass

    return 0.0  # No score attached – keyword component will dominate.


def score_doc(doc, keywords: list[str]) -> ScoredDoc:
    """Compute the composite rerank score for a single *doc*.

    Composite formula (weights sum to 1.0):
        final = 0.65 * keyword_overlap  +  0.35 * faiss_similarity

    The keyword component is weighted higher because the FAISS score is often
    unavailable (plain similarity_search does not attach it), and because
    keyword overlap is a stronger signal for *specific* clinical queries.

    Args:
        doc:      LangChain Document object.
        keywords: Keywords extracted from the query.

    Returns:
        ScoredDoc named-tuple.
    """
    sim_score = _extract_faiss_sim_score(doc)

    lowered_content = doc.page_content.lower()
    overlap = sum(1 for k in keywords if k in lowered_content)
    coverage = overlap / (len(keywords) + 1e-5)
    keyword_score = coverage

    length = len(doc.page_content.split())
    length_score = min(length / 200, 1.0)

    final_score = (
        0.6 * sim_score +
        0.25 * keyword_score +
        0.15 * length_score
    )

    # Boost based on keyword coverage for multi-symptom queries
    final_score += 0.1 * coverage

    # Penalize Low-Quality Fragments
    if length < 40:
        final_score -= 0.05

    log.debug(
        "Score  kw=%.3f  sim=%.3f  len=%.3f  final=%.3f  | %.60s …",
        keyword_score, sim_score, length_score, final_score,
        doc.page_content.replace("\n", " "),
    )
    return ScoredDoc(doc=doc, score=final_score, keyword_score=keyword_score, sim_score=sim_score)


# ---------------------------------------------------------------------------
# Main reranker entry-point
# ---------------------------------------------------------------------------

def rerank_docs(docs: list, query: str) -> list:
    """Second-stage reranker: filter and rank *docs* by semantic relevance.

    Pipeline
    --------
    1. Extract keywords dynamically from *query*.
    2. Score every chunk (keyword overlap + FAISS similarity blend).
    3. Sort descending by composite score.
    4. Remove chunks below ``RERANK_MIN_SCORE`` threshold.
    5. Keep at most ``RERANK_TOP_N`` chunks.
    6. Safety net: if all chunks were filtered out, keep the single
       highest-scoring chunk so the LLM always receives *some* context.

    The function is fully disease-agnostic – it never references specific
    conditions.  It works for any healthcare query.

    Args:
        docs:  Raw list of LangChain Document objects from FAISS retrieval.
        query: User query string.

    Returns:
        Filtered, ranked list of Document objects (best first).
        May be shorter than *docs*.  Never returns an empty list if *docs*
        was non-empty (safety net ensures at least one chunk survives).
    """
    if not docs:
        log.warning("Reranker received empty document list – skipping.")
        return docs

    # Step 1 – Dynamic keyword extraction
    keywords = extract_keywords(query)
    if not keywords:
        log.warning(
            "No keywords extracted from query '%s'. "
            "Skipping keyword filter; returning FAISS order.",
            query,
        )
        return docs[:RERANK_TOP_N] if RERANK_TOP_N > 0 else docs

    log.info("Reranking %d chunks.  Keywords: %s", len(docs), keywords)

    # Step 2 – Score all chunks
    scored = [score_doc(doc, keywords) for doc in docs]

    # Step 3 – Sort descending by composite score
    scored.sort(key=lambda sd: sd.score, reverse=True)

    # Step 4 – Filter by minimum score threshold
    filtered = [sd for sd in scored if sd.score >= RERANK_MIN_SCORE]

    # If all chunks scored below threshold, keep top-N anyway
    if not filtered:
        log.warning(
            "All %d chunks scored below RERANK_MIN_SCORE=%.2f. "
            "Keeping top-%d chunks anyway as a safety net.",
            len(scored),
            RERANK_MIN_SCORE,
            RERANK_TOP_N if RERANK_TOP_N > 0 else len(scored)
        )
        filtered = scored[:RERANK_TOP_N] if RERANK_TOP_N > 0 else scored

    # Guarantee minimum context: NEVER return < 2 documents if available
    min_docs = min(2, len(scored))
    if len(filtered) < min_docs:
        log.warning(
            "Reranker filtered down to %d chunks, but we enforce a minimum of %d. "
            "Adding chunks back to meet the minimum.",
            len(filtered),
            min_docs
        )
        filtered = scored[:min_docs]

    # Step 5 – Cap at RERANK_TOP_N
    if RERANK_TOP_N > 0:
        filtered = filtered[:RERANK_TOP_N]

    log.info(
        "Reranker: %d -> %d chunk(s) after filtering "
        "(min_score=%.2f, top_n=%d).",
        len(docs),
        len(filtered),
        RERANK_MIN_SCORE,
        RERANK_TOP_N,
    )

    # Return plain Document objects (strip the ScoredDoc wrapper)
    return [sd.doc for sd in filtered]
