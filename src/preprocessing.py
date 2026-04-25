"""
preprocessing.py  –  Text refinement layer for the Clinical RAG pipeline.

Role in the pipeline
---------------------
    data_loader.py          -->  list[str]  (Patient/Doctor dialogue strings)
         |
    preprocessing.py        -->  list[str]  (cleaned, deduplicated, length-safe)
         |
    embedder.py             -->  FAISS vector store

This module does NOT load any data from disk.  It receives the list of
already-formatted documents produced by ``data_loader.load_clinical_documents``
and applies a series of lightweight text-refinement steps to improve
retrieval quality and embedding performance.

No heavy NLP libraries are used.  All transformations rely on Python's
standard library (``re``, ``unicodedata``) to keep the module fast and
dependency-free.

Public API
----------
    from src.preprocessing import preprocess_documents
    clean_docs = preprocess_documents(raw_docs)

Each helper is also importable individually for unit testing.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from src.config import (
    PREPROCESSING_LOWERCASE,
    PREPROCESSING_MAX_WORDS,
    PREPROCESSING_MIN_WORDS,
)
from src.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Compiled patterns  (compiled once at import time for speed)
# ---------------------------------------------------------------------------

# Any run of whitespace that is NOT a plain newline: tabs, carriage returns,
# non-breaking spaces, zero-width characters, multiple spaces, etc.
_INLINE_WS_RE = re.compile(
    r"[ \t\r\f\v\u00a0\u200b\u200c\u200d\u2060\ufeff]+"
)

# More than one consecutive newline  →  single newline
_MULTI_NL_RE = re.compile(r"\n{2,}")

# Characters that are clearly artefacts in medical text scraped from the web:
# bullet dots, dashes used as list markers, pipe separators, repeated dashes.
_ARTIFACT_RE = re.compile(
    r"(?:"
    r"\u2022|\u2023|\u25aa|\u25cf"   # bullet variants
    r"|(?<!\w)-{2,}(?!\w)"           # standalone dash runs  (not mid-word)
    r"|\|{2,}"                        # double-or-more pipe
    r"|\#{2,}"                        # double hash (not clinical markdown)
    r")"
)

# The canonical section headers produced by data_loader._format_document.
# We use these to detect and normalise the boundary between patient and doctor.
_PATIENT_HEADER_RE = re.compile(r"^Patient\s*:\s*", re.IGNORECASE)
_DOCTOR_HEADER_RE  = re.compile(r"^Doctor\s*:\s*",  re.IGNORECASE)


# ---------------------------------------------------------------------------
# Step 1 – Unicode normalisation
# ---------------------------------------------------------------------------

def normalise_unicode(text: str) -> str:
    """Normalise Unicode to NFC form and replace fancy punctuation.

    Converts curly quotes, em-dashes, and other typographic characters to
    their plain ASCII equivalents so the embedding model sees consistent
    tokens.  NFC normalisation also resolves composed vs decomposed forms.

    Args:
        text: Raw input string.

    Returns:
        NFC-normalised string with typographic characters replaced.

    Example:
        >>> normalise_unicode("\u201cHello\u201d")
        '"Hello"'
    """
    # NFC: canonical composition (e.g. é as one code point, not e + combining)
    text = unicodedata.normalize("NFC", text)

    # Curly / smart quotes  →  straight quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")   # '' -> '
    text = text.replace("\u201c", '"').replace("\u201d", '"')   # "" -> "

    # Em-dash / en-dash  →  hyphen (keeps meaning in clinical ranges: "3–5 days")
    text = text.replace("\u2014", "-").replace("\u2013", "-")

    # Horizontal ellipsis  →  three dots
    text = text.replace("\u2026", "...")

    return text


# ---------------------------------------------------------------------------
# Step 2 – Whitespace normalisation
# ---------------------------------------------------------------------------

def normalise_whitespace(text: str) -> str:
    """Collapse irregular whitespace while preserving paragraph structure.

    Two transforms are applied:

    1. Any run of non-newline whitespace (tabs, multiple spaces, NBSP, etc.)
       is replaced with a single ASCII space.
    2. More than one consecutive newline is collapsed to exactly one,
       preserving the ``Patient / Doctor`` boundary without adding blank lines.

    Args:
        text: Input string (may contain tabs, NBSP, repeated spaces).

    Returns:
        Whitespace-normalised string with leading / trailing space stripped.

    Example:
        >>> normalise_whitespace("Hello   world\\n\\n\\nHow are you?")
        'Hello world\\nHow are you?'
    """
    text = _INLINE_WS_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Step 3 – Artefact removal
# ---------------------------------------------------------------------------

def remove_artifacts(text: str) -> str:
    """Remove web-scraping artefacts that add noise without adding meaning.

    Targets bullet characters, run-on dashes used as list separators, and
    stray pipe characters.  Conservative: only removes characters that are
    clearly formatting noise and carry no medical meaning.

    Does NOT touch:
    - Single hyphens (used in compound words and dosage ranges).
    - Single pipes (rare in clinical text but possible in tables).
    - Any alphabetic or numeric content.

    Args:
        text: Input string possibly containing bullet/dash artefacts.

    Returns:
        String with formatting artefacts removed and whitespace renormalised.
    """
    text = _ARTIFACT_RE.sub(" ", text)
    # Re-run whitespace normalisation to collapse any gaps left behind.
    return normalise_whitespace(text)


# ---------------------------------------------------------------------------
# Step 4 – Section header consistency
# ---------------------------------------------------------------------------

def fix_section_headers(text: str) -> str:
    """Enforce canonical casing and spacing for Patient / Doctor headers.

    ``data_loader`` always writes ``Patient: …\\nDoctor: …``, but edge cases
    (e.g. extra spaces, mixed case from older dataset versions) can slip
    through.  This function enforces the canonical form so the retriever
    always sees a consistent prefix.

    Canonical form::

        Patient: <question text>
        Doctor: <answer text>

    Args:
        text: A document string, possibly with irregular header formatting.

    Returns:
        Document with headers normalised to canonical form.
    """
    lines = text.split("\n")
    fixed: list[str] = []

    for line in lines:
        if _PATIENT_HEADER_RE.match(line):
            # Strip existing header and rewrite canonically.
            body = _PATIENT_HEADER_RE.sub("", line).strip()
            line = f"Patient: {body}"
        elif _DOCTOR_HEADER_RE.match(line):
            body = _DOCTOR_HEADER_RE.sub("", line).strip()
            line = f"Doctor: {body}"
        fixed.append(line)

    return "\n".join(fixed)


# ---------------------------------------------------------------------------
# Step 5 – Optional lowercase normalisation
# ---------------------------------------------------------------------------

def maybe_lowercase(text: str, apply: bool) -> str:
    """Optionally convert text to lowercase.

    Lowercase normalisation can improve recall for keyword-heavy queries but
    may harm precision for proper nouns (drug names, gene names).  It is
    therefore **disabled by default** and controlled via the
    ``PREPROCESSING_LOWERCASE`` environment variable.

    Args:
        text:  Input string.
        apply: If ``True``, return ``text.lower()``; otherwise return as-is.

    Returns:
        Optionally lowercased string.
    """
    return text.lower() if apply else text


# ---------------------------------------------------------------------------
# Step 6 – Length checks
# ---------------------------------------------------------------------------

def count_words(text: str) -> int:
    """Return the number of whitespace-separated tokens in *text*.

    Args:
        text: Any string.

    Returns:
        Integer word count.
    """
    return len(text.split())


def is_too_short(text: str, min_words: int) -> bool:
    """Return ``True`` if *text* contains fewer than *min_words* words.

    Documents that are near-empty (e.g. a header with no body) contribute
    noise rather than signal to the vector store.

    Args:
        text:      Document string.
        min_words: Minimum acceptable word count (inclusive).

    Returns:
        ``True`` if the document should be discarded.
    """
    return count_words(text) < min_words


def trim_to_max_words(text: str, max_words: int) -> str:
    """Truncate *text* to at most *max_words* words.

    Truncation is a last-resort safety measure: very long documents can
    exceed the embedding model's context window and degrade representation
    quality.  A hard truncation is safer than silently producing a bad
    embedding.

    The function preserves whole words; it never cuts in the middle of a
    token.  The caller is responsible for deciding whether truncation is
    appropriate for their use case (controlled via ``PREPROCESSING_MAX_WORDS``
    in ``.env``; set to 0 to disable).

    Args:
        text:      Input string.
        max_words: Maximum number of words to keep (0 = no limit).

    Returns:
        Original string if it is within the limit; otherwise a truncated copy.
    """
    if max_words <= 0:
        return text

    words = text.split()
    if len(words) <= max_words:
        return text

    return " ".join(words[:max_words])


# ---------------------------------------------------------------------------
# Step 7 – Deduplication
# ---------------------------------------------------------------------------

def deduplicate(documents: list[str]) -> tuple[list[str], int]:
    """Remove exact-duplicate documents while preserving insertion order.

    Uses a ``set`` of seen documents for O(1) membership checks.  Only
    exact string matches are considered duplicates; near-duplicate detection
    (e.g. Jaccard similarity) is deliberately out of scope to keep the
    module lightweight.

    Args:
        documents: List of document strings (order is preserved for uniques).

    Returns:
        A tuple of:
        - ``list[str]``: deduplicated documents in original order.
        - ``int``:       number of duplicates that were removed.
    """
    seen: set[str] = set()
    unique: list[str] = []

    for doc in documents:
        if doc not in seen:
            seen.add(doc)
            unique.append(doc)

    removed = len(documents) - len(unique)
    return unique, removed


# ---------------------------------------------------------------------------
# Step 8 – Single-document pipeline
# ---------------------------------------------------------------------------

def _process_single(
    doc: str,
    lowercase: bool,
    max_words: int,
) -> Optional[str]:
    """Apply the full refinement pipeline to one document.

    This private helper is called by :func:`preprocess_documents` for each
    document.  Returning ``None`` signals that the document should be dropped.

    Pipeline order
    ~~~~~~~~~~~~~~
    1. Unicode normalisation (resolve composed forms, replace fancy punct).
    2. Artefact removal (bullets, run-on dashes).
    3. Whitespace normalisation (collapse spaces/newlines).
    4. Section header consistency (canonicalise Patient / Doctor prefix).
    5. Optional lowercase.
    6. Max-word truncation (if configured).
    7. Return ``None`` if the document is now below the minimum word count.

    Args:
        doc:       Raw document string from ``data_loader``.
        lowercase: Whether to apply lowercase normalisation.
        max_words: Maximum word count (0 = no limit).

    Returns:
        Refined document string, or ``None`` if the document should be dropped.
    """
    doc = normalise_unicode(doc)
    doc = remove_artifacts(doc)
    doc = normalise_whitespace(doc)
    doc = fix_section_headers(doc)
    doc = maybe_lowercase(doc, lowercase)
    doc = trim_to_max_words(doc, max_words)

    # Drop documents that shrank below the minimum after all transforms.
    if is_too_short(doc, PREPROCESSING_MIN_WORDS):
        return None

    return doc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_documents(documents: list[str]) -> list[str]:
    """Refine a list of formatted clinical documents before embedding.

    This is the **single entry point** for this module.  Call it inside
    ``main.py`` (or any other orchestrator) immediately after
    ``data_loader.load_clinical_documents`` returns, and pass the result
    directly to the embedder::

        docs = load_clinical_documents()
        docs = preprocess_documents(docs)
        build_vector_store(docs)

    What this function does
    ~~~~~~~~~~~~~~~~~~~~~~~
    For each document (in order):

    1. Normalises Unicode and replaces typographic punctuation.
    2. Strips web-scraping artefacts (bullets, run-on dashes).
    3. Collapses irregular whitespace without breaking line structure.
    4. Enforces canonical ``Patient: / Doctor:`` headers.
    5. Optionally converts to lowercase (off by default; set
       ``PREPROCESSING_LOWERCASE=true`` in ``.env`` to enable).
    6. Optionally truncates to ``PREPROCESSING_MAX_WORDS`` words
       (0 = disabled).
    7. Drops documents shorter than ``PREPROCESSING_MIN_WORDS`` words.

    After per-document processing, exact duplicates are removed.

    Medical content safety
    ~~~~~~~~~~~~~~~~~~~~~~
    No stemming, stopword removal, or NLP tokenisation is applied.  Drug
    names, dosages, and clinical terminology are preserved verbatim.

    Args:
        documents: ``list[str]`` of ``Patient: …\\nDoctor: …`` strings
                   returned by ``data_loader.load_clinical_documents``.

    Returns:
        ``list[str]`` of refined documents, ready to be passed to the
        embedding step.  Empty if all input documents were invalid or
        duplicate.

    Raises:
        TypeError: If *documents* is not a list.
    """
    if not isinstance(documents, list):
        raise TypeError(
            f"preprocess_documents expects a list[str], got {type(documents).__name__}."
        )

    input_count = len(documents)
    log.info("Preprocessing started. Input documents: %d", input_count)
    log.info(
        "Settings  |  lowercase=%s  max_words=%d  min_words=%d",
        PREPROCESSING_LOWERCASE,
        PREPROCESSING_MAX_WORDS,
        PREPROCESSING_MIN_WORDS,
    )

    # --- Per-document transforms ---
    processed: list[str] = []
    dropped_short = 0
    modified = 0

    for original in documents:
        refined = _process_single(
            original,
            lowercase=PREPROCESSING_LOWERCASE,
            max_words=PREPROCESSING_MAX_WORDS,
        )

        if refined is None:
            dropped_short += 1
            continue

        if refined != original:
            modified += 1

        processed.append(refined)

    after_cleaning = len(processed)

    # --- Deduplication ---
    processed, dropped_dupe = deduplicate(processed)
    final_count = len(processed)

    # --- Summary log ---
    log.info("-" * 52)
    log.info("Preprocessing complete.")
    log.info("  Input documents   : %d", input_count)
    log.info("  Modified          : %d", modified)
    log.info("  Dropped (too short): %d", dropped_short)
    log.info("  Dropped (duplicate): %d", dropped_dupe)
    log.info("  Output documents  : %d", final_count)
    log.info("-" * 52)

    if final_count == 0:
        log.warning(
            "All %d documents were removed during preprocessing. "
            "Review PREPROCESSING_MIN_WORDS and your input data.",
            input_count,
        )

    return processed
