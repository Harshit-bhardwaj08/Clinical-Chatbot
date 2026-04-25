"""
MediChat RAG Pipeline: A robust, grounded engine for clinical Q&A.

This module implements several layers of reliability to ensure we don't 
crash or hallucinate:
- We use adaptive TOP-K retrieval to avoid memory issues with Ollama.
- A grounding validator ensures every claim is backed by retrieved docs.
- We have a custom confidence scorer to warn users when evidence is thin.

If you need to see what's happening under the hood, you can activate
DEBUG_MODE in your .env file to see detailed pipeline stats in the console.
"""

from __future__ import annotations

import re
import time
import math
from typing import Literal, Optional

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from src.config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    TOP_K_RESULTS,
    MAX_CONTEXT_CHARS,
    FALLBACK_TOP_K_MEDIUM,
    FALLBACK_TOP_K_MIN,
    LLM_TIMEOUT_SECONDS,
    DEBUG_MODE,
)
from src.reranker import rerank_docs, extract_keywords, score_doc
from src.logger import get_logger

log = get_logger(__name__)

ConfidenceLevel = Literal["low", "medium", "high"]
QualityLevel = Literal["rich", "partial", "weak"]


# ── Embeddings & Similiarity ──
# We cache embeddings to speed up repetitive grounding checks.

_EMBEDDING_MODEL = None
_EMBEDDING_CACHE = {}

def get_embedding(text: str) -> list[float]:
    """Get embedding vector using the shared global model with simple caching."""
    global _EMBEDDING_MODEL
    global _EMBEDDING_CACHE
    
    if text in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[text]
        
    if _EMBEDDING_MODEL is None:
        from src.embedder import get_embedding_model
        _EMBEDDING_MODEL = get_embedding_model()
        
    emb = _EMBEDDING_MODEL.embed_query(text)
    _EMBEDDING_CACHE[text] = emb
    return emb

def compute_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


# ── Grounding Thresholds ──

_GROUNDING_NGRAM_THRESHOLD: float = 0.40
_GROUNDING_MIN_WORDS: int = 6
_NGRAM_SIZE: int = 3


# ── The Clinical Prompt ──
# This prompt is strictly tuned to prevent hallucinations and enforce 
# a professional medical tone.

_SYSTEM_HEADER = """\
You are MediChat, a clinical medical assistant. Answer questions clearly and professionally.

FORMATTING RULES — follow exactly:

1. DEFINITION QUESTIONS ("what is X", "define X", "tell me about X"):
   Respond with exactly 1-2 sentences. Bold the **Condition Name**. Do NOT list symptoms or treatments. Example:
   **Typhoid fever** is a bacterial infection caused by Salmonella typhi, spread through contaminated food and water.

2. SYMPTOM QUESTIONS ("what are symptoms", "signs of", "how does it present"):
   First line MUST be: Symptoms of **[Actual Condition Name]** include:
   Then list EACH symptom on its OWN LINE starting with a dash and space: - Symptom name
   Last line: If symptoms persist or worsen, seek medical attention.
   Example format:
   Symptoms of **Typhoid fever** include:
   - High fever
   - Weakness and fatigue
   - Headache
   - Stomach pain
   - Loss of appetite
   If symptoms persist or worsen, seek medical attention.

3. TREATMENT / CAUSES / PREVENTION / OTHER:
   Write 1-2 sentence summary, then list details with: - point
   Bold all **key medical terms**.

CRITICAL RULES:
- ALWAYS use the REAL condition name from the context (e.g., "Typhoid fever", not "This condition" or "the condition").
- For follow-up questions, resolve "it", "its symptoms", "the disease" using chat history.
- NEVER say: "According to", "Based on the context", "the provided context", "passage", "source", "records".
- NEVER echo or repeat these instructions.
- NEVER add section headers like "FORMATTING INSTRUCTIONS" or "REMINDER" in your answer.
"""

_PROMPT_TEMPLATE = """{system}

--- MEDICAL CONTEXT ---
{context}
--- END CONTEXT ---

--- CONVERSATION HISTORY ---
{chat_history}
--- END HISTORY ---

Question: {question}
{dynamic_constraint}
Answer:"""


# ---------------------------------------------------------------------------
# Vague context signals
# ---------------------------------------------------------------------------

_VAGUE_PHRASES: tuple[str, ...] = (
    "non-specific",
    "nonspecific",
    "unclear",
    "cannot determine",
    "not sure",
    "vague",
    "uncertain",
    "no clear",
    "limited information",
    "insufficient",
    "unable to determine",
)


_SYMPTOM_TERMS: tuple[str, ...] = (
    "symptom",
    "symptoms",
    "sign",
    "signs",
    "present",
    "presentation",
)

_DEFINITION_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^what\s+is\s+", re.IGNORECASE),
    re.compile(r"^what\'s\s+", re.IGNORECASE),
    re.compile(r"^define\s+", re.IGNORECASE),
    re.compile(r"^tell\s+me\s+about\s+", re.IGNORECASE),
    re.compile(r"^explain\s+", re.IGNORECASE),
)

_CONDITION_FROM_QUESTION: tuple[re.Pattern, ...] = (
    re.compile(r"what\s+is\s+(.+?)\??$", re.IGNORECASE),
    re.compile(r"what\'s\s+(.+?)\??$", re.IGNORECASE),
    re.compile(r"define\s+(.+?)\??$", re.IGNORECASE),
    re.compile(r"tell\s+me\s+about\s+(.+?)\??$", re.IGNORECASE),
    re.compile(r"symptoms?\s+of\s+(.+?)\??$", re.IGNORECASE),
    re.compile(r"signs?\s+of\s+(.+?)\??$", re.IGNORECASE),
)

_CONDITION_FROM_DOC = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[a-z]+){0,3}\s+(?:fever|disease|syndrome|infection|cancer))\b"
)

_GENERIC_BULLET_TERMS: set[str] = {
    "symptom",
    "symptoms",
    "condition",
    "disease",
    "this condition",
    "this disease",
}


def _classify_question_intent(question: str) -> str:
    """Classify question intent into definition, symptoms, or other."""
    q = (question or "").strip().lower()
    if any(p.search(q) for p in _DEFINITION_PATTERNS):
        return "definition"
    if any(term in q for term in _SYMPTOM_TERMS):
        return "symptoms"
    return "other"


def _normalise_condition_name(name: str) -> str:
    """Normalize condition candidate text for display."""
    cleaned = re.sub(r"\s+", " ", (name or "").strip(" .,:;!?\n\t"))
    cleaned = re.sub(r"^(a|an|the)\s+", "", cleaned, flags=re.IGNORECASE)
    if not cleaned:
        return ""
    if cleaned.islower():
        return cleaned[0].upper() + cleaned[1:]
    return cleaned


def _extract_condition_from_question(question: str) -> str:
    """Extract condition name from the current question, when explicit."""
    q = (question or "").strip()
    for pattern in _CONDITION_FROM_QUESTION:
        m = pattern.search(q)
        if m:
            return _normalise_condition_name(m.group(1))
    return ""


def _extract_condition_from_history(history: Optional[list[dict]]) -> str:
    """Fallback condition extraction from previous user turns."""
    if not history:
        return ""

    for msg in reversed(history):
        if msg.get("role") != "user":
            continue
        candidate = _extract_condition_from_question(msg.get("content", ""))
        if candidate:
            return candidate
    return ""


def _extract_condition_from_docs(docs: list) -> str:
    """Best-effort condition extraction from retrieved context."""
    for doc in docs:
        content = getattr(doc, "page_content", "")
        m = _CONDITION_FROM_DOC.search(content)
        if m:
            return _normalise_condition_name(m.group(1))
    return ""


def _resolve_condition_name(question: str, history: Optional[list[dict]], docs: list) -> str:
    """Resolve condition name using question, chat history, then retrieved docs."""
    for resolver in (
        lambda: _extract_condition_from_question(question),
        lambda: _extract_condition_from_history(history),
        lambda: _extract_condition_from_docs(docs),
    ):
        value = resolver()
        if value:
            return value
    return ""


def _non_bullet_lines(answer: str) -> list[str]:
    """Return non-empty lines excluding markdown bullet lines."""
    lines: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^[-*•]\s+", stripped):
            continue
        lines.append(stripped)
    return lines


def _extract_sentences(text: str) -> list[str]:
    """Split text into readable sentences."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _clean_bullet_text(item: str, condition: str = "") -> str:
    """Remove noisy and generic bullet entries."""
    cleaned = re.sub(r"\s+", " ", item.strip(" -•*\t\n\r"))
    cleaned = re.sub(r"^[0-9]+[\).]\s*", "", cleaned)
    cleaned = cleaned.strip(" .,:;!")
    if not cleaned:
        return ""
    lower = cleaned.lower()
    if lower in _GENERIC_BULLET_TERMS:
        return ""
    if condition and lower == condition.lower():
        return ""
    if condition:
        tokens = {t for t in re.findall(r"[a-z]+", condition.lower()) if len(t) > 3}
        if lower in tokens:
            return ""
    return cleaned


def _extract_bullets(answer: str, condition: str = "") -> list[str]:
    """Extract bullets from markdown lines and inline symptom phrases."""
    bullets: list[str] = []

    for line in answer.splitlines():
        stripped = line.strip()
        m = re.match(r"^[-*•]\s+(.+)$", stripped)
        if m:
            point = _clean_bullet_text(m.group(1), condition)
            if point:
                bullets.append(point)

    # Inline dash list patterns like "summary. - point A - point B"
    inline_dash = re.findall(r"(?:^|\s)-\s+([^\n-][^-]+?)(?=\s+-\s+|$)", answer)
    for item in inline_dash:
        point = _clean_bullet_text(item, condition)
        if point:
            bullets.append(point)

    # Inline patterns like "Symptoms include: fever, headache, fatigue"
    inline_matches = re.findall(
        r"(?:include|includes|are)\s*:\s*([^\n.]+)",
        answer,
        flags=re.IGNORECASE,
    )
    for part in inline_matches:
        for chunk in re.split(r",|;", part):
            point = _clean_bullet_text(chunk, condition)
            if point:
                bullets.append(point)

    seen: set[str] = set()
    unique: list[str] = []
    for point in bullets:
        key = point.lower()
        if key not in seen:
            seen.add(key)
            unique.append(point)
    return unique


def _extract_symptoms_from_docs(docs: list, condition: str = "") -> list[str]:
    """Fallback symptom extraction from retrieved docs when LLM list is poor."""
    bullets: list[str] = []
    symptom_triggers = re.compile(
        r"symptoms?\s+(?:include|are)|signs?\s+(?:include|are)",
        re.IGNORECASE,
    )

    for doc in docs:
        content = getattr(doc, "page_content", "")
        for sentence in _extract_sentences(content):
            if not symptom_triggers.search(sentence):
                continue
            if ":" in sentence:
                tail = sentence.split(":", 1)[-1]
            else:
                tail = re.sub(
                    r"^.*?(symptoms?|signs?)\s+(include|are)\s+",
                    "",
                    sentence,
                    flags=re.IGNORECASE,
                )
            for item in re.split(r",|;| and ", tail):
                point = _clean_bullet_text(item, condition)
                if point:
                    bullets.append(point)
            if len(bullets) >= 10:
                break
        if len(bullets) >= 10:
            break

    seen: set[str] = set()
    unique: list[str] = []
    for point in bullets:
        key = point.lower()
        if key not in seen:
            seen.add(key)
            unique.append(point)
    return unique


def _format_definition_answer(condition: str, answer: str) -> str:
    """Ensure definition answers stay concise and clean."""
    lines = _non_bullet_lines(answer)
    text = " ".join(lines)
    sentences = _extract_sentences(text)
    summary = " ".join(sentences[:2]).strip() if sentences else text.strip()

    # Drop malformed symptom-style lines when user asked for a definition.
    if re.search(r"symptoms?.*include", summary, flags=re.IGNORECASE) or "this condition" in summary.lower():
        summary = ""

    if not summary:
        if condition:
            return f"**{condition}** is a medical condition that should be evaluated clinically."
        return "This is a medical condition that should be evaluated clinically."

    if condition and f"**{condition}**" not in summary and condition.lower() not in summary.lower():
        summary = f"**{condition}** is {summary[0].lower() + summary[1:] if len(summary) > 1 else summary.lower()}"
    elif condition and f"**{condition}**" not in summary:
        summary = re.sub(
            re.escape(condition),
            f"**{condition}**",
            summary,
            count=1,
            flags=re.IGNORECASE,
        )

    return summary.strip()


def _format_symptom_answer(condition: str, answer: str, docs: list) -> str:
    """Build a stable symptom response with heading + bullets + safety line."""
    bullets = _extract_bullets(answer, condition=condition)
    if len(bullets) < 3:
        fallback = _extract_symptoms_from_docs(docs, condition=condition)
        for point in fallback:
            if point.lower() not in {b.lower() for b in bullets}:
                bullets.append(point)
            if len(bullets) >= 10:
                break

    if not bullets:
        prose = " ".join(_non_bullet_lines(answer))
        for sentence in _extract_sentences(prose):
            for item in re.split(r",|;| and ", sentence):
                point = _clean_bullet_text(item, condition=condition)
                if point:
                    bullets.append(point)
            if len(bullets) >= 10:
                break

    if not bullets:
        cleaned = answer.strip()
        if cleaned:
            return cleaned
        return "I could not extract a reliable symptom list from the retrieved records."

    heading = (
        f"Symptoms of **{condition}** include:"
        if condition
        else "Symptoms include:"
    )

    symptom_block = "\n".join(f"- {item}" for item in bullets[:10])
    tail = "If symptoms persist or worsen, seek medical attention."
    return f"{heading}\n\n{symptom_block}\n\n{tail}"


def _format_general_answer(answer: str) -> str:
    """Ensure a readable summary-first structure for general questions."""
    bullets = _extract_bullets(answer)
    prose_lines = _non_bullet_lines(answer)
    prose_text = " ".join(prose_lines).strip()
    prose_sentences = _extract_sentences(prose_text)

    if not bullets:
        return answer.strip()

    summary = " ".join(prose_sentences[:2]).strip() if prose_sentences else ""
    if not summary:
        summary = "Here are the key points:"

    bullet_block = "\n".join(f"- {point}" for point in bullets[:10])
    return f"{summary}\n\n{bullet_block}".strip()


def _shape_answer_for_intent(
    question: str,
    answer: str,
    docs: list,
    history: Optional[list[dict]] = None,
) -> str:
    """Post-process answer into predictable, intent-aware markdown structure."""
    intent = _classify_question_intent(question)
    condition = _resolve_condition_name(question, history, docs)

    if intent == "definition":
        return _format_definition_answer(condition, answer)
    if intent == "symptoms":
        return _format_symptom_answer(condition, answer, docs)
    return _format_general_answer(answer)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_llm() -> OllamaLLM:
    """Instantiate the Ollama LLM with a request timeout."""
    log.info(
        "Initialising Ollama  (model=%s, url=%s, timeout=%ss).",
        OLLAMA_MODEL,
        OLLAMA_BASE_URL,
        LLM_TIMEOUT_SECONDS,
    )
    return OllamaLLM(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        timeout=LLM_TIMEOUT_SECONDS,
    )


def _build_prompt() -> PromptTemplate:
    """Build the clinical PromptTemplate."""
    return PromptTemplate(
        template=_PROMPT_TEMPLATE,
        input_variables=["system", "context", "chat_history", "question", "dynamic_constraint"],
    )


def _retrieve_docs(retriever, question: str, k: int) -> list:
    """Fetch up to *k* relevant documents for *question*."""
    try:
        retriever.search_kwargs["k"] = k
    except (AttributeError, TypeError):
        pass

    try:
        return retriever.invoke(question)
    except AttributeError:
        return retriever.get_relevant_documents(question)


def _deduplicate_docs(docs: list) -> list:
    """Remove exact-duplicate chunks, preserving retrieval rank."""
    seen: set[str] = set()
    unique: list = []
    for doc in docs:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            unique.append(doc)
    return unique


def _enforce_diversity(chunks: list, threshold: float = 0.85) -> tuple[list, int]:
    """Ensure semantic diversity by filtering out redundant chunks using embeddings."""
    if not chunks:
        return [], 0
        
    selected_chunks = []
    selected_embeddings = []
    removed_count = 0
    
    for chunk in chunks:
        emb = get_embedding(chunk.page_content.strip())
        
        is_similar = False
        for existing_emb in selected_embeddings:
            if compute_similarity(emb, existing_emb) > threshold:
                is_similar = True
                break
                
        if not is_similar:
            selected_chunks.append(chunk)
            selected_embeddings.append(emb)
        else:
            removed_count += 1
            
    # Guarantee minimum 2 chunks always retained (fallback rule)
    if len(selected_chunks) < 2 and len(chunks) >= 2:
        selected_chunks = chunks[:2]
        removed_count = len(chunks) - len(selected_chunks)
            
    return selected_chunks, removed_count


def _enforce_context_budget(docs: list, budget_chars: int) -> tuple[list, int]:
    """Trim docs until combined context fits within budget_chars."""
    if budget_chars <= 0:
        total = sum(len(d.page_content) for d in docs)
        return docs, total

    kept = list(docs)
    while kept:
        total = sum(len(d.page_content.strip()) for d in kept)
        if total <= budget_chars:
            return kept, total
        longest_idx = max(range(len(kept)), key=lambda i: len(kept[i].page_content))
        dropped = kept.pop(longest_idx)
        log.debug(
            "Context budget: dropped chunk of %d chars (budget=%d, current=%d).",
            len(dropped.page_content),
            budget_chars,
            total,
        )

    return [], 0


def _format_context(docs: list) -> str:
    """Produce a clean context block from docs without indices."""
    if not docs:
        return "No relevant context found."
    # We strip indices to prevent the LLM from referencing "Passage [1]"
    parts = [doc.page_content.strip() for doc in docs]
    return "\n\n---\n\n".join(parts)


def _call_llm(llm: OllamaLLM, filled_prompt: str) -> str:
    """Call Ollama and return a clean response, preserving all markdown structure."""
    try:
        log.info("Calling LLM...")
        response = llm.invoke(filled_prompt)

        # 1. Strip numeric citation artifacts like [1], [2], [1,2]
        response = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', response)

        # 2. Strip lines that echo system prompt instructions
        _ECHO_PATTERNS = re.compile(
            r'^(if the question|formatting (rules|instructions)|critical rules|'
            r'do not (write|repeat|echo|explain)|for follow-up|line 1 must|'
            r'then each symptom|final line:|write \d|bold (the|all)|stop immediately|'
            r'write a \d|definition questions|symptom questions|treatment.*causes|'
            r'reminder:|note:|answer:|example format:)',
            re.IGNORECASE
        )
        lines = response.split('\n')
        lines = [ln for ln in lines if not _ECHO_PATTERNS.match(ln.strip())]
        response = '\n'.join(lines)

        # 3. Remove meta-talk openers referencing context/source/passages
        _META_TALK = re.compile(
            r'(according to (the )?(provided |given )?(context|passages?|records?|information|sources?|dataset)[^.]*\.\s*)'
            r'|(based on (the )?(provided |given )?(context|passages?|records?|information|sources?|dataset)[^.]*\.\s*)'
            r'|(from (the )?(provided |given )?(context|passages?|information|sources?)[^.]*\.\s*)',
            re.IGNORECASE
        )
        response = _META_TALK.sub('', response)

        # 4. Strip leftover "---" separator artifacts from context blocks
        response = re.sub(r'\n?---+\n?', '\n', response)

        # 5. Fix inline/comma-separated symptom lists (e.g., "- Fever, Headache, Pain") → split to individual bullets
        def expand_comma_bullets(m):
            items = [item.strip() for item in m.group(1).split(',') if item.strip()]
            return '\n'.join(f'- {item}' for item in items)
        response = re.sub(r'^-\s+([A-Z][a-zA-Z\s]+(?:,\s*[A-Z][a-zA-Z\s]+)+)$', expand_comma_bullets, response, flags=re.MULTILINE)

        # 6. Fix inline bullets appended after sentence: "text. - Next" → split
        response = re.sub(r'([.?!])\s+-\s+(?=[A-Z*])', r'\1\n- ', response)

        # 7. Handle bullets written with • or * instead of -
        response = re.sub(r'^[•\*]\s+', '- ', response, flags=re.MULTILINE)

        # 8. Ensure tight bullet lists:
        # First, ensure exactly ONE newline before every bullet
        response = re.sub(r'\n+-\s+', '\n- ', response)
        
        # Then, ensure exactly TWO newlines BEFORE the FIRST bullet in a list
        # This is needed for Streamlit to recognize the start of a markdown list.
        # We look for a line that is NOT a bullet, followed by a bullet.
        response = re.sub(r'([^\n])\n(- )', r'\1\n\n\2', response)

        # 9. Collapse 3+ blank lines to max 2
        response = re.sub(r'\n{3,}', '\n\n', response)

        return response.strip()
    except Exception as e:
        log.error("LLM call failed: %s", str(e))
        return "I'm sorry, I encountered an error while processing your request."


# ---------------------------------------------------------------------------
# Grounding Validator & Hallucination Control
# ---------------------------------------------------------------------------

def _get_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    """Convert text into a set of lowercase n-grams (tuples of words)."""
    words = [w.lower() for w in re.findall(r"\w+", text)]
    if len(words) < n:
        return set()
    return set(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def _sentence_is_grounded(sentence: str, context_text: str) -> bool:
    """Return True if the sentence is sufficiently supported by the context."""
    s_strip = sentence.strip()

    # Short fragments (headers, single words) are kept — not enough text to
    # make a falsifiable claim, and grounding would produce false positives.
    words = sentence.split()
    if len(words) < _GROUNDING_MIN_WORDS:
        return True

    s_ngrams = _get_ngrams(sentence, _NGRAM_SIZE)
    if not s_ngrams:
        return True

    c_ngrams = _get_ngrams(context_text, _NGRAM_SIZE)
    overlap = s_ngrams.intersection(c_ngrams)

    ratio = len(overlap) / len(s_ngrams)

    # Keyword safety net: if at least one meaningful word from this sentence
    # also appears in the context, we keep the sentence even with low n-gram overlap.
    sentence_words = {w.lower() for w in re.findall(r"\w+", sentence) if len(w) > 3}
    context_words = {w.lower() for w in re.findall(r"\w+", context_text) if len(w) > 3}
    has_keyword_match = bool(sentence_words.intersection(context_words))

    # Only strip a sentence when n-gram overlap is very low AND no keyword match.
    if ratio < 0.2 and not has_keyword_match:
        return False

    return True


def _validate_grounding(raw_answer: str, docs: list) -> tuple[str, int]:
    """Remove sentences from the LLM output that lack n-gram support in the docs.
    
    IMPORTANT: Preserves the original line-break structure of the answer so that
    markdown bullet points render correctly in Streamlit.
    """
    if not docs:
        return raw_answer, 0

    combined_context = " ".join(d.page_content for d in docs)
    
    # --- Split by LINE first to preserve markdown structure ---
    lines = raw_answer.split('\n')
    kept_lines: list[str] = []
    removed_count = 0
    
    for line in lines:
        stripped = line.strip()

        # Always keep blank lines — they control paragraph spacing in markdown.
        if not stripped:
            kept_lines.append(line)
            continue

        # Always keep header-like lines that end with ':' (e.g. "Symptoms of X include:")
        if stripped.endswith(':'):
            kept_lines.append(line)
            continue

        # Bullet lines are now subject to keyword grounding.
        # A bullet with zero overlap with any context keyword is flagged.
        if stripped.startswith(('-', '*', '•')):
            bullet_text = re.sub(r'^[-*\u2022]\s*', '', stripped)
            if _sentence_is_grounded(bullet_text, combined_context):
                kept_lines.append(line)
            else:
                removed_count += 1
                log.debug("Grounding failure (bullet). Removed: '%s'", stripped)
            continue

        # For prose lines, split into sentences and check each one individually.
        sub_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', stripped)
        line_kept_parts: list[str] = []
        
        for s in sub_sentences:
            if not s.strip():
                continue
            if _sentence_is_grounded(s, combined_context):
                line_kept_parts.append(s)
            else:
                removed_count += 1
                log.debug("Grounding failure. Removed: '%s'", s.strip())
        
        if line_kept_parts:
            kept_lines.append(' '.join(line_kept_parts))
    
    # EMPTY ANSWER FALLBACK
    result = '\n'.join(kept_lines).strip()
    if not result:
        bullets = []
        for d in docs:
            d_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", d.page_content)
            for ds in d_sentences:
                if len(ds.split()) > 5:
                    bullets.append(f"- {ds.strip()}")
                    if len(bullets) >= 3:
                        break
            if len(bullets) >= 3:
                break
        fallback_answer = "Here are the relevant clinical points:\n\n" + "\n".join(bullets)
        log.info("Grounding fallback triggered: returning %d extracted points", len(bullets))
        return fallback_answer, len(lines)

    log.info("Grounding complete: kept %d/%d lines", len([l for l in kept_lines if l.strip()]), len([l for l in lines if l.strip()]))
    return result, removed_count


def _get_removed_sentences(raw_answer: str, validated_answer: str) -> list[str]:
    """Helper to figure out exactly which strings were stripped."""
    raw_s = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9*])", raw_answer)
    val_s = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9*])", validated_answer)
    return [s for s in raw_s if s not in val_s]


# ---------------------------------------------------------------------------
# Confidence & Quality Assessment
# ---------------------------------------------------------------------------

def _chunk_has_substantive_claims(chunk_text: str) -> bool:
    """Return True if chunk contains dense medical terminology or specific numbers."""
    text = chunk_text.lower()
    has_numbers = bool(re.search(r"\d", text))

    medical_words = {
        "treatment", "symptom", "diagnosis", "disease", "syndrome",
        "therapy", "medication", "dose", "mg", "chronic", "acute",
        "pain", "blood", "pressure", "glucose", "insulin", "patient",
        "risk", "factor", "surgery", "infection", "virus", "bacteria",
        "cancer", "tumor", "heart", "brain", "liver", "kidney",
    }

    words = set(re.findall(r"\w+", text))
    med_word_count = len(words.intersection(medical_words))
    return has_numbers or med_word_count >= 3


def _assess_context_quality(
    docs: list, question: str, context_query: str = ""
) -> tuple[ConfidenceLevel, QualityLevel, float, int, int, int, str]:
    """Score the context and determine the confidence level."""
    if not docs:
        return "low", "weak", 0.0, 0, 0, 0, "No context returned from retriever."

    # Use context_query (topic + question) for better keyword detection in follow-ups
    eval_query = context_query if context_query else question
    keywords = extract_keywords(eval_query)
    combined_text = " ".join(d.page_content.lower() for d in docs)

    hits = sum(1 for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", combined_text))
    total_kw = len(keywords)
    density = hits / total_kw if total_kw > 0 else 0.0
    substantive_count = sum(1 for d in docs if _chunk_has_substantive_claims(d.page_content))

    if density >= 0.75 and substantive_count >= 1:
        quality: QualityLevel = "rich"
        conf: ConfidenceLevel = "high"
        reason = f"Excellent keyword coverage ({hits}/{total_kw}) with substantive claims."
    elif density >= 0.40 or (density > 0 and substantive_count >= 1):
        quality = "partial"
        conf = "medium"
        reason = f"Partial keyword coverage ({hits}/{total_kw}). May lack specifics."
    else:
        quality = "weak"
        conf = "low"
        reason = f"Poor keyword coverage ({hits}/{total_kw}). Context seems irrelevant."

    return conf, quality, density, hits, total_kw, substantive_count, reason


def _apply_weak_context_preamble(answer: str, confidence: ConfidenceLevel) -> str:
    """Ensures the answer isn't a flat refusal if some info exists."""
    # We now rely fully on the LLM to integrate 'limited info' warnings gracefully.
    return answer


# ---------------------------------------------------------------------------
# Core Pipeline Logic
# ---------------------------------------------------------------------------

class _RobustRAGChain:
    """The main RAG pipeline executor."""

    def __init__(self, vector_store, prompt: PromptTemplate, llm: OllamaLLM, top_k_steps: list[int]):
        self.vector_store = vector_store
        self.prompt = prompt
        self.llm = llm
        self.top_k_steps = top_k_steps

    def run(self, question: str, history: list[dict] = None) -> dict:
        """Execute the pipeline with fallback logic."""
        last_error = None
        log.info(f"Fallback schedule: {self.top_k_steps}")
        for attempt_num, top_k in enumerate(self.top_k_steps, 1):
            log.info(f"Creating retriever (top_k={top_k})")
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": top_k}
            )
            try:
                return self._attempt(retriever, question, top_k, attempt_num, history)
            except Exception as e:  # noqa: BLE001
                last_error = e
                log.warning(
                    "[Attempt %d] Pipeline failed: %s. "
                    "Will fallback to smaller k if available.",
                    attempt_num, e,
                )
                time.sleep(1.0)

        log.error("All RAG attempts failed. Last error: %s", last_error)
        return {
            "answer": "I am currently unable to process your request due to an internal system error.",
            "sources": [],
            "context_chars": 0,
            "top_k_used": 0,
            "confidence": "low",
            "grounding_removals": 0,
            "diagnostic_report": None,
        }

    def _attempt(self, retriever, question: str, k: int, attempt_num: int, history: list[dict] = None) -> dict:
        """Run a single iteration of the RAG pipeline."""
        
        history = history or []
        history_str = ""
        if history:
            for msg in history[-5:]: # Keep last 5 turns to save context budget
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "")
                history_str += f"{role}: {content}\n"
        else:
            history_str = "No prior history."
        
        # Enhance retrieval query with history context
        search_query = question
        if history:
            user_msgs = [msg["content"] for msg in history if msg.get("role") == "user"]
            if user_msgs:
                # If question is vague/short, use more history. 
                # If it's specific, just use the last turn for context.
                if len(question.split()) < 5:
                    active_topic = " ".join(user_msgs[-2:])
                    search_query = f"{active_topic} {question}"
                else:
                    search_query = f"{user_msgs[-1]} {question}"
        
        # Step 1 - Retrieve expanded candidate pool
        raw_docs = _retrieve_docs(retriever, search_query, k)
        log.info("Initial retrieval: %d chunks using query: %s", len(raw_docs), search_query)

        if DEBUG_MODE:
            print("\n" + "="*40 + " DEBUG MODE " + "="*40)
            print(f"Query: {question}")
            print(f"Retrieval: Found {len(raw_docs)} chunks from FAISS.")
            
            # PHASE 1: Root Cause Verification
            if len(raw_docs) > 1:
                highly_similar = 0
                embs = [get_embedding(d.page_content.strip()) for d in raw_docs]
                comparisons = 0
                for i in range(len(raw_docs)):
                    for j in range(i + 1, len(raw_docs)):
                        comparisons += 1
                        if compute_similarity(embs[i], embs[j]) > 0.85:
                            highly_similar += 1
                
                print(f"Semantic redundancy: {highly_similar}/{comparisons} chunks are highly similar")

        # Step 2 - Deduplicate
        docs_before_dedup = raw_docs
        docs = _deduplicate_docs(docs_before_dedup)
        dedup_removed = len(docs_before_dedup) - len(docs)
        
        if DEBUG_MODE:
            print(f"Deduplication: Removed {dedup_removed} duplicates.")

        # Step 2.5 - Semantic Diversity Filter (Phase 2)
        docs_before_diversity = len(docs)
        docs, diversity_removed = _enforce_diversity(docs, threshold=0.85)
        
        log.info("After diversity: %d -> %d chunks", docs_before_diversity, len(docs))
        if DEBUG_MODE:
            print(f"After diversity: {docs_before_diversity} -> {len(docs)} chunks")

        # Step 3 - Rerank
        docs_before_rerank = len(docs)
        docs = rerank_docs(docs, question)
        
        # Truncate to final top_k
        docs = docs[:k]
        
        log.info("After rerank: %d -> %d chunks", docs_before_rerank, len(docs))
        if DEBUG_MODE:
            print(f"After rerank: {docs_before_rerank} -> {len(docs)} chunks")

        # Step 4 - Context budget
        docs, context_chars = _enforce_context_budget(docs, MAX_CONTEXT_CHARS)
        log.info(
            "[Attempt %d]  After budget enforcement: %d chunk(s), %d chars.",
            attempt_num, len(docs), context_chars,
        )

        # Step 5 - Confidence
        (
            confidence,
            quality_class,
            kw_density,
            kw_hits,
            kw_total,
            substantive_count,
            confidence_reason,
        ) = _assess_context_quality(docs, question, context_query=search_query)

        # Step 6 - Build prompt
        context = _format_context(docs)
        
        dynamic_constraint = ""
        if quality_class == "weak":
            dynamic_constraint = (
                "\nNOTE: Provide only the facts you found. Do NOT explain that information is missing "
                "or that the context is limited. Just give the best answer possible with the available data.\n"
            )
            
        filled_prompt = self.prompt.format(
            system=_SYSTEM_HEADER,
            context=context,
            chat_history=history_str,
            question=question,
            dynamic_constraint=dynamic_constraint,
        )
        log.info(
            "[Attempt %d]  Prompt size: %d chars. Sending to Ollama ...",
            attempt_num, len(filled_prompt),
        )

        if DEBUG_MODE:
            print(f"Context Quality: {quality_class.upper()} | Confidence: {confidence.upper()}")
            print(f"Reason: {confidence_reason}")

        # Step 7 - LLM call (raises on failure, caught by run())
        raw_answer = _call_llm(self.llm, filled_prompt)

        # Step 8 - Grounding validation
        validated_answer, removals = _validate_grounding(raw_answer.strip(), docs)

        if DEBUG_MODE:
            print(f"Hallucination Control: {removals} sentences stripped.")
            if removals > 0:
                removed_sentences = _get_removed_sentences(raw_answer.strip(), validated_answer)
                for i, s in enumerate(removed_sentences):
                    print(f"  - Removed [{i+1}]: {s[:80]}...")
            print("="*92 + "\n")

        # Step 9 - Intent-aware answer shaping
        shaped_answer = _shape_answer_for_intent(
            question=question,
            answer=validated_answer,
            docs=docs,
            history=history,
        )

        # Step 10 - Weak-context preamble
        final_answer = _apply_weak_context_preamble(shaped_answer, confidence)

        return {
            "answer":             final_answer,
            "sources":            [doc.page_content for doc in docs],
            "context_chars":      context_chars,
            "top_k_used":         k,
            "confidence":         confidence,
            "grounding_removals": removals,
            "diagnostic_report":  None,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_rag_chain(vector_store) -> _RobustRAGChain:
    """Construct the RAG pipeline and return a chain object."""
    llm    = _build_llm()
    prompt = _build_prompt()

    fallback_schedule = [
        TOP_K_RESULTS,
        FALLBACK_TOP_K_MEDIUM,
        FALLBACK_TOP_K_MIN
    ]
    top_k_steps = sorted(set(k for k in fallback_schedule if k > 0), reverse=True)

    log.info(
        "RAG chain ready (v3). Fallback schedule: %s | budget: %d chars | timeout: %ss.",
        top_k_steps,
        MAX_CONTEXT_CHARS,
        LLM_TIMEOUT_SECONDS,
    )
    return _RobustRAGChain(vector_store, prompt, llm, top_k_steps)


def query(chain: _RobustRAGChain, question: str, history: list[dict] = None) -> dict:
    """Run a question through the RAG chain."""
    if not question or not question.strip():
        return {
            "answer": "Please provide a non-empty question.",
            "sources": [],
            "context_chars": 0,
            "top_k_used": 0,
            "confidence": "low",
            "grounding_removals": 0,
        }

    log.info("Received query (%d chars).", len(question))
    result = chain.run(question.strip(), history=history)
    log.info(
        "Query complete. top_k_used=%d, context=%d chars, answer=%d chars, "
        "confidence=%s, grounding_removals=%d.",
        result["top_k_used"],
        result["context_chars"],
        len(result["answer"]),
        result["confidence"],
        result["grounding_removals"],
    )
    return result
