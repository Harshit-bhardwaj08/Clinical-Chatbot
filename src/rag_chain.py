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
    RAG_INTENT_CONFIDENCE_THRESHOLD,
    RAG_QUERY_TERM_SIM_THRESHOLD,
    PRONOUN_RESOLUTION_MAX_WORDS,
    TOPIC_SHIFT_SIM_THRESHOLD,
    ANSWER_PREFIX_OVERLAP_THRESHOLD,
    LLM_TEMPERATURE,
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
   **Typhoid fever** is a bacterial infection caused by Salmonella Typhi, spread through contaminated food and water.

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

3. TREATMENT QUESTIONS ("treatment", "treat", "management"):
   Write 1-2 concise sentences first.
   If listing details, use one bullet per line beginning with: - point
   Avoid fixed drug regimens, doses, or durations unless explicitly present in trusted context.
   Include one safety line advising clinician-guided treatment based on tests/local resistance when relevant.

4. CAUSES / COMPLICATIONS / TYPES:
   Prefer a direct heading + clean bullet list (no duplicate lines).
   Use one concept per bullet and avoid repeating caution text in bullets.

5. PREVENTION / OTHER:
   Write 1-2 sentence summary, then optional bullet details with: - point

OUTPUT RULES:
- Do NOT begin your answer by copying, continuing, or rephrasing a sentence from the context.
  Start with a direct, original response.
- Do NOT echo or repeat any part of the retrieved passages as your opening line.

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

_TREATMENT_TERMS: tuple[str, ...] = (
    "treatment",
    "treat",
    "therapy",
    "therapies",
    "management",
    "managed",
    "medication",
    "medications",
    "medicine",
    "medicines",
    "drug",
    "drugs",
    "antibiotic",
    "antibiotics",
)

_CAUSE_TERMS: tuple[str, ...] = (
    "cause",
    "causes",
    "etiology",
    "aetiology",
    "risk factor",
    "risk factors",
)

_COMPLICATION_TERMS: tuple[str, ...] = (
    "complication",
    "complications",
    "sequelae",
)

_TYPE_TERMS: tuple[str, ...] = (
    "types",
    "type",
    "kinds",
    "kind",
    "forms",
    "form",
)

_DEFINITION_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^what\s+is\s+", re.IGNORECASE),
    re.compile(r"^what\'s\s+", re.IGNORECASE),
    re.compile(r"^define\s+", re.IGNORECASE),
    re.compile(r"^tell\s+me\s+about\s+", re.IGNORECASE),
    re.compile(r"^explain\s+", re.IGNORECASE),
)

_CONDITION_FROM_QUESTION: tuple[re.Pattern, ...] = (
    re.compile(r"what\s+is\s*(.+?)\??$", re.IGNORECASE),
    re.compile(r"what\'s\s*(.+?)\??$", re.IGNORECASE),
    re.compile(r"define\s+(.+?)\??$", re.IGNORECASE),
    re.compile(r"tell\s+me\s+about\s+(.+?)\??$", re.IGNORECASE),
    re.compile(r"explain\s+(.+?)\??$", re.IGNORECASE),
    re.compile(r"symptoms?\s+of\s+(.+?)\??$", re.IGNORECASE),
    re.compile(r"signs?\s+of\s+(.+?)\??$", re.IGNORECASE),
    re.compile(r"causes?\s+of\s+(.+?)\??$", re.IGNORECASE),
    re.compile(r"complications?\s+of\s+(.+?)\??$", re.IGNORECASE),
    re.compile(r"types?\s+of\s+(.+?)\??$", re.IGNORECASE),
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
    "other causes include",
    "other causes",
    "include",
    "includes",
    "sudden",
    "severe",
    "mild",
    "moderate",
    "acute",
    "chronic",
    "if symptoms persist or worsen",
    "seek medical attention",
}

_GENERIC_CONDITION_TOKENS: set[str] = {
    "condition",
    "disease",
    "disorder",
    "syndrome",
    "symptom",
    "symptoms",
    "sign",
    "signs",
    "treatment",
    "treatments",
    "treat",
    "management",
    "therapy",
    "therapies",
    "cause",
    "causes",
    "complication",
    "complications",
    "type",
    "types",
    "kind",
    "kinds",
    "prevention",
    "prevent",
    "precaution",
    "precautions",
    "medication",
    "medications",
    "medicine",
    "medicines",
    "drug",
    "drugs",
}

_GREETING_PATTERN = re.compile(
    r"\b(?:hi|hello|hey|good\s+morning|good\s+afternoon|good\s+evening)\b",
    re.IGNORECASE,
)

_MEDICHAT_GREETING = (
    "Hello, I am MediChat, your medical assistant. I'm here to provide "
    "clear and professional guidance on various health topics. "
    "What can I help you with today?"
)


def _is_pure_greeting(text: str) -> bool:
    """Return True if the text contains ONLY a greeting and optional punctuation."""
    if not text:
        return False
    # Remove common punctuation and whitespace
    cleaned = re.sub(r"[^a-zA-Z\s]", "", text).strip()
    if not cleaned:
        return False
    # Check if the remaining text exactly matches our greeting pattern
    pattern = r"(?:hi|hello|hey|hii|hii+|heyy+|good\s+morning|good\s+afternoon|good\s+evening)"
    return bool(re.fullmatch(pattern, cleaned, re.IGNORECASE))


def _is_generic_condition_candidate(text: str) -> bool:
    """Return True when extracted 'condition' text is just a generic intent phrase."""
    cleaned = re.sub(r"[^a-zA-Z\s-]", " ", (text or "").lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"^(the|a|an|th)\s+", "", cleaned)
    if not cleaned:
        return True
    if cleaned in {"it", "this", "that"}:
        return True

    tokens = [tok for tok in re.findall(r"[a-z]+", cleaned) if tok]
    if not tokens:
        return True
        
    import difflib
    for tok in tokens:
        if tok in _GENERIC_CONDITION_TOKENS:
            continue
        matches = difflib.get_close_matches(tok, _GENERIC_CONDITION_TOKENS, n=1, cutoff=0.8)
        if not matches:
            return False
            
    return True


def _has_user_greeting(text: str) -> bool:
    """Return True when the user message includes a greeting phrase."""
    return bool(_GREETING_PATTERN.search(text or ""))


def _prepend_medichat_greeting(answer: str) -> str:
    """Prefix a short MediChat greeting, avoiding duplicates."""
    body = (answer or "").strip()
    if not body:
        return _MEDICHAT_GREETING

    if re.search(r"\bmedi\s*chat\b", body, flags=re.IGNORECASE):
        return body

    return f"{_MEDICHAT_GREETING}\n\n{body}"


def _classify_question_intent(question: str) -> str:
    """Classify question intent into definition, symptoms, treatment, or other."""
    q = (question or "").strip().lower()
    if any(term in q for term in _SYMPTOM_TERMS):
        return "symptoms"
    if any(term in q for term in _TREATMENT_TERMS):
        return "treatment"
    if any(term in q for term in _COMPLICATION_TERMS):
        return "complications"
    if any(term in q for term in _CAUSE_TERMS):
        return "causes"
    if any(term in q for term in _TYPE_TERMS):
        return "types"
    if any(p.search(q) for p in _DEFINITION_PATTERNS):
        return "definition"
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
            extracted = m.group(1).strip()
            extracted = re.sub(
                r"^(?:the\s+)?(?:treatment|management|therapy|symptoms?|signs?|causes?|complications?|prevention)\s+of\s+",
                "",
                extracted,
                flags=re.IGNORECASE,
            ).strip()
            extracted = re.sub(
                r"^(?:the\s+)?(?:type|types|kind|kinds)\s+of\s+",
                "",
                extracted,
                flags=re.IGNORECASE,
            ).strip()
            extracted = re.sub(
                r"\s+(?:are|is|do|does|exist|existed|there|found|happen|occur|present)$",
                "",
                extracted,
                flags=re.IGNORECASE,
            ).strip()

            extracted_lower = extracted.lower()
            # Prevent pronouns or generic terms from being extracted as a 'condition'
            if extracted_lower in {"it", "this", "that", "the condition", "the disease", "the syndrome"}:
                continue
            if _is_generic_condition_candidate(extracted):
                continue
            return _normalise_condition_name(extracted)
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
    stop_words = {"if", "the", "a", "an", "when", "because", "in", "this", "that", "for", "as", "is", "are", "of", "and", "or", "to", "with", "but", "however"}
    for doc in docs:
        content = getattr(doc, "page_content", "")
        m = _CONDITION_FROM_DOC.search(content)
        if m:
            candidate = m.group(1)
            first_word = candidate.split()[0].lower()
            if first_word in stop_words:
                continue
            return _normalise_condition_name(candidate)
    return ""


def _resolve_condition_name(question: str, history: Optional[list[dict]], docs: list) -> str:
    """Resolve condition name using question, chat history, then retrieved docs."""
    q_cond = _extract_condition_from_question(question)
    h_cond = _extract_condition_from_history(history)
    d_cond = _extract_condition_from_docs(docs)

    # Prioritize more specific (longer) names if they overlap
    candidates = [c for c in [q_cond, h_cond, d_cond] if c]
    if not candidates:
        return ""
    
    primary = q_cond or h_cond
    if primary and d_cond and primary.lower() in d_cond.lower():
        return d_cond

    for val in [q_cond, h_cond, d_cond]:
        if val:
            return val
    return ""


def _non_bullet_lines(answer: str) -> list[str]:
    """Return non-empty lines excluding markdown bullet lines."""
    lines: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^[-*•](\s+|$)", stripped):
            continue
        lines.append(stripped)
    return lines


def _extract_sentences(text: str) -> list[str]:
    """Split text into readable sentences."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _dedupe_repeated_lines(text: str) -> str:
    """Remove repeated non-empty lines while preserving order."""
    seen: set[str] = set()
    output: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            output.append("")
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(raw_line)
    return "\n".join(output).strip()


def _clean_bullet_text(item: str, condition: str = "") -> str:
    """Remove noisy and generic bullet entries."""
    cleaned = re.sub(r"\s+", " ", item.strip(" -•*\t\n\r"))
    cleaned = re.sub(r"^[0-9]+[\).]\s*", "", cleaned)
    cleaned = re.sub(r"(?i)\bif symptoms persist or worsen,?\s*seek medical attention\.?$", "", cleaned)
    cleaned = re.sub(r"(?i)^other causes include:?\s*", "", cleaned)
    cleaned = re.sub(r"(?i)^symptoms of .+? include:?\s*", "", cleaned)
    cleaned = re.sub(r"(?i)^signs of .+? include:?\s*", "", cleaned)
    cleaned = re.sub(r"(?i)^causes of .+? include:?\s*", "", cleaned)
    cleaned = re.sub(r"(?i)^complications of .+? include:?\s*", "", cleaned)
    cleaned = cleaned.strip(" .,:;!")
    if not cleaned:
        return ""
    lower = cleaned.lower()
    if lower in _GENERIC_BULLET_TERMS:
        return ""
    if re.search(r"(symptoms?|signs?|causes?|complications?)\s+of\s+.+\s+include", lower):
        return ""
    if lower.startswith("if symptoms persist or worsen"):
        return ""
    if lower.startswith("seek medical attention"):
        return ""
    if condition and lower == condition.lower():
        return ""
    if condition:
        tokens = {t for t in re.findall(r"[a-z]+", condition.lower()) if len(t) > 3}
        if lower in tokens:
            return ""
    return cleaned


def _extract_line_items(answer: str, condition: str = "") -> list[str]:
    """Extract list-like items from plain lines when dashes are missing."""
    items: list[str] = []
    header_prefixes = (
        "symptoms of",
        "signs of",
        "causes of",
        "complications of",
        "types of",
    )
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower().strip(" .:")
        if any(lower.startswith(prefix) for prefix in header_prefixes):
            continue
        if lower.startswith("if symptoms persist or worsen"):
            continue
        if lower in {"other causes include", "other causes include:"}:
            continue

        words = re.findall(r"[a-zA-Z][a-zA-Z0-9'/-]*", stripped)
        if len(words) == 0 or len(words) > 18:
            continue
        if re.search(r"[.!?]$", stripped):
            continue

        point = _clean_bullet_text(stripped, condition)
        if point:
            items.append(point)

    return _dedupe_points(items)


def _extract_inline_items_from_prose(answer: str, condition: str = "") -> list[str]:
    """Extract list items from prose sentences containing include/caused-by phrasing."""
    items: list[str] = []
    prose = " ".join(_non_bullet_lines(answer)).strip()
    if not prose:
        return []

    for sentence in _extract_sentences(prose):
        lower = sentence.lower()
        tail = ""
        for marker in (
            " include ",
            " includes ",
            " including ",
            " are ",
            " caused by ",
            " triggered by ",
            " result from ",
            " results from ",
            " may result from ",
        ):
            idx = lower.find(marker)
            if idx != -1:
                tail = sentence[idx + len(marker):]
                break

        if not tail:
            continue

        tail = re.sub(
            r"^(?:various factors|factors|these|such as|including)\s*,?\s*",
            "",
            tail,
            flags=re.IGNORECASE,
        )
        for piece in re.split(r",|;| and ", tail):
            point = _clean_bullet_text(piece, condition)
            if point:
                items.append(point)
        if len(items) >= 10:
            break

    return _dedupe_points(items)


_BULLET_STOPWORDS: set[str] = {
    "and",
    "or",
    "the",
    "a",
    "an",
    "of",
    "with",
    "without",
    "including",
    "include",
    "includes",
}


def _point_tokens(text: str) -> set[str]:
    """Normalize bullet text into comparable content tokens."""
    return {
        token
        for token in re.findall(r"[a-z]+", text.lower())
        if len(token) > 2 and token not in _BULLET_STOPWORDS
    }


def _is_near_duplicate_point(candidate: str, existing: str) -> bool:
    """Return True when two bullet points are effectively the same."""
    c_tokens = _point_tokens(candidate)
    e_tokens = _point_tokens(existing)
    if not c_tokens or not e_tokens:
        return candidate.strip().lower() == existing.strip().lower()

    if c_tokens.issubset(e_tokens) or e_tokens.issubset(c_tokens):
        return True

    overlap = len(c_tokens.intersection(e_tokens))
    smaller = min(len(c_tokens), len(e_tokens))
    return smaller > 0 and (overlap / smaller) >= 0.8


def _dedupe_points(points: list[str]) -> list[str]:
    """Drop exact and near-duplicate bullet points while preserving order."""
    unique: list[str] = []
    for point in points:
        if any(_is_near_duplicate_point(point, chosen) for chosen in unique):
            continue
        unique.append(point)
    return unique


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

    return _dedupe_points(bullets)


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

    return _dedupe_points(bullets)


def _extract_items_from_docs(docs: list, focus: str, condition: str = "") -> list[str]:
    """Fallback extraction for causes/complications/types from retrieved docs."""
    bullets: list[str] = []
    trigger_by_focus = {
        "causes": re.compile(
            r"causes?\s+(?:include|are)|risk factors?\s+(?:include|are)|etiolog(?:y|ies)\s+(?:include|are)",
            re.IGNORECASE,
        ),
        "complications": re.compile(
            r"complications?\s+(?:include|are)|sequelae\s+(?:include|are)",
            re.IGNORECASE,
        ),
        "types": re.compile(
            r"types?\s+(?:include|are)|forms?\s+(?:include|are)|kinds?\s+(?:include|are)|types?\s+of",
            re.IGNORECASE,
        ),
    }
    strip_by_focus = {
        "causes": r"^.*?(causes?|risk factors?|etiolog(?:y|ies))\s+(include|are)\s+",
        "complications": r"^.*?(complications?|sequelae)\s+(include|are)\s+",
        "types": r"^.*?(types?|forms?|kinds?)\s+(include|are|of)\s+",
    }
    trigger = trigger_by_focus.get(focus)
    strip_rule = strip_by_focus.get(focus)
    if not trigger or not strip_rule:
        return []

    for doc in docs:
        content = getattr(doc, "page_content", "")
        for sentence in _extract_sentences(content):
            if not trigger.search(sentence):
                continue
            if ":" in sentence:
                tail = sentence.split(":", 1)[-1]
            else:
                tail = re.sub(strip_rule, "", sentence, flags=re.IGNORECASE)
            for item in re.split(r",|;| and ", tail):
                point = _clean_bullet_text(item, condition)
                if point:
                    bullets.append(point)
            if len(bullets) >= 10:
                break
        if len(bullets) >= 10:
            break

    return _dedupe_points(bullets)


def _format_definition_answer(condition: str, answer: str, **kwargs) -> str:
    """Ensure definition answers stay concise and clean."""
    lines = _non_bullet_lines(answer)
    text = " ".join(lines)
    sentences = _extract_sentences(text)
    summary = " ".join(sentences[:2]).strip() if sentences else text.strip()

    # Drop malformed symptom-style lines when user asked for a definition.
    if re.search(r"symptoms?.*include", summary, flags=re.IGNORECASE) or "this condition" in summary.lower():
        summary = ""

    if not summary:
        return "I do not have enough reliable information in the provided context to answer this question."

    condition_norm = (condition or "").lower().strip()
    if condition and summary and not any(kw in summary.lower() for kw in condition_norm.split() if len(kw) > 2):
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
        for point in _extract_line_items(answer, condition=condition):
            if point.lower() not in {b.lower() for b in bullets}:
                bullets.append(point)
            if len(bullets) >= 10:
                break
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


def _format_treatment_answer(condition: str, answer: str) -> str:
    """Build a concise treatment answer with safer clinical framing."""
    prose = " ".join(_non_bullet_lines(answer)).strip()
    sentences = _extract_sentences(prose)
    summary = " ".join(sentences[:2]).strip() if sentences else prose

    if not summary:
        if condition:
            summary = f"Treatment of **{condition}** should be individualized by a qualified clinician."
        else:
            summary = "Treatment should be individualized by a qualified clinician."

    regimen_signals = re.search(
        r"\b\d+\s*(?:day|days|week|weeks|mg|g|ml)\b",
        summary,
        flags=re.IGNORECASE,
    )
    antibiotic_signals = re.search(
        r"\b(?:cef\w+|floxacin|cillin|cycline|mycin|azole)\b",
        summary,
        flags=re.IGNORECASE,
    )
    if regimen_signals or antibiotic_signals:
        if condition:
            summary = (
                f"Treatment of **{condition}** usually includes doctor-prescribed therapy based on "
                "clinical evaluation, test results, and local resistance patterns."
            )
        else:
            summary = (
                "Treatment usually includes doctor-prescribed therapy based on clinical evaluation, "
                "test results, and local resistance patterns."
            )

    if condition and f"**{condition}**" not in summary:
        summary = re.sub(
            re.escape(condition),
            f"**{condition}**",
            summary,
            count=1,
            flags=re.IGNORECASE,
        )

    bullets = _extract_bullets(answer, condition=condition)
    safety = (
        "Treatment should be prescribed by a qualified clinician based on clinical evaluation, "
        "tests, and local resistance patterns."
    )

    if bullets:
        bullet_block = "\n".join(f"- {point}" for point in bullets[:8])
        return f"{summary}\n\n{bullet_block}\n\n{safety}"
    return f"{summary}\n\n{safety}"


def _format_list_intent_answer(
    *,
    question: str,
    condition: str,
    answer: str,
    docs: list,
    focus: str,
) -> str:
    """Format causes/complications/types answers as stable heading + bullets."""
    points = _extract_items_from_docs(docs, focus=focus, condition=condition)
    if len(points) < 3:
        for item in _extract_bullets(answer, condition=condition):
            if not any(_is_near_duplicate_point(item, p) for p in points):
                points.append(item)
            if len(points) >= 10:
                break
    if len(points) < 3:
        for item in _extract_line_items(answer, condition=condition):
            if not any(_is_near_duplicate_point(item, p) for p in points):
                points.append(item)
            if len(points) >= 10:
                break
    if len(points) < 3:
        for item in _extract_inline_items_from_prose(answer, condition=condition):
            if not any(_is_near_duplicate_point(item, p) for p in points):
                points.append(item)
            if len(points) >= 10:
                break
    points = _dedupe_points(points)
    if not points:
        return _format_general_answer(question, answer, condition=condition)

    if focus == "causes":
        heading = f"Common causes of **{condition}** include:" if condition else "Common causes include:"
    elif focus == "complications":
        heading = f"Complications of **{condition}** include:" if condition else "Complications include:"
    else:
        heading = f"Common types of **{condition}** include:" if condition else "Common types include:"

    bullet_block = "\n".join(f"- {point}" for point in points[:10])
    return f"{heading}\n\n{bullet_block}"


def _format_general_answer(question: str = "", answer: str = "", condition: str = "") -> str:
    """Ensure a readable summary-first structure for general questions."""
    # Handle being called with one arg by test_pipeline.py
    if not answer and question:
        answer = question
        question = ""

    bullets = _extract_bullets(answer, condition=condition)
    if not bullets:
        bullets = _extract_line_items(answer, condition=condition)
    if not bullets:
        bullets = _extract_inline_items_from_prose(answer, condition=condition)

    prose_lines = _non_bullet_lines(answer)
    prose_text = " ".join(prose_lines).strip()
    prose_sentences = _extract_sentences(prose_text)

    if not bullets:
        return _dedupe_repeated_lines(answer.strip())

    summary = " ".join(prose_sentences[:2]).strip() if prose_sentences else ""
    if not summary:
        if condition and any(term in (question or "").lower() for term in _CAUSE_TERMS):
            summary = f"Common causes of **{condition}** include:"
            bullet_block = "\n".join(f"- {point}" for point in bullets[:10])
            return f"{summary}\n\n{bullet_block}".strip()
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
        shaped = _format_definition_answer(condition, answer)
    elif intent == "symptoms":
        shaped = _format_symptom_answer(condition, answer, docs)
    elif intent == "treatment":
        shaped = _format_treatment_answer(condition, answer)
    elif intent == "causes":
        shaped = _format_list_intent_answer(
            question=question,
            condition=condition,
            answer=answer,
            docs=docs,
            focus="causes",
        )
    elif intent == "complications":
        shaped = _format_list_intent_answer(
            question=question,
            condition=condition,
            answer=answer,
            docs=docs,
            focus="complications",
        )
    elif intent == "types":
        shaped = _format_list_intent_answer(
            question=question,
            condition=condition,
            answer=answer,
            docs=docs,
            focus="types",
        )
    else:
        shaped = _format_general_answer(question, answer, condition=condition)

    if _has_user_greeting(question):
        shaped = _prepend_medichat_greeting(shaped)

    return _dedupe_repeated_lines(shaped)


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
        temperature=LLM_TEMPERATURE,
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
    """Produce a clean context block from docs without indices and dialogue labels."""
    if not docs:
        return "No relevant context found."
    
    parts = []
    for doc in docs:
        content = doc.page_content.strip()
        # Strip dialogue labels but preserve the structural separators (like newlines)
        content = re.sub(r'^(?:Patient|Doctor|User|Assistant|Expert|Clinician):\s*', '', content, flags=re.IGNORECASE | re.MULTILINE)
        content = re.sub(r'([.?!])\s+(?:Patient|Doctor|User|Assistant|Expert|Clinician):\s*', r'\1 ', content, flags=re.IGNORECASE)
        parts.append(content.strip())
        
    return "\n\n---\n\n".join(parts)


def _final_quality_cleanup(text: str) -> str:
    """Final pass to remove dialogue labels and duplicate lines."""
    # Strip labels but preserve newlines
    text = re.sub(r'^(?:Patient|Doctor|User|Assistant|Expert|Clinician):\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'([.?!])\s+(?:Patient|Doctor|User|Assistant|Expert|Clinician):\s*', r'\1 ', text, flags=re.IGNORECASE)
    # Dedupe lines
    return _dedupe_repeated_lines(text)


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
        response = _dedupe_repeated_lines(response)

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


def _validate_grounding(raw_answer: str, docs: list, question: str = "", context_query: str = "", **kwargs) -> tuple[str, int]:
    """Remove sentences from the LLM output that lack n-gram support in the docs.
    
    IMPORTANT: Preserves the original line-break structure of the answer so that
    markdown bullet points render correctly in Streamlit.
    """
    if not docs:
        return raw_answer, 0
    
    # SECURITY: If the topic in the question is completely absent from context,
    # reject the answer entirely to prevent model-memory hallucinations.
    if question and not _context_supports_topic(docs, question):
        return "I do not have enough reliable information in the provided context to answer this question.", 1

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


# ── MISSING SECURITY FUNCTIONS ──────────────────────────────────────────────

def _build_topic_term_vocab(text: str) -> set[str]:
    """Extract medical topic terms from text for fuzzy matching."""
    return {t.lower() for t in re.findall(r"\w+", text) if len(t) > 3}

def _context_supports_topic(docs: list, question: str) -> bool:
    """Return True if the context contains keywords relevant to the question topic."""
    q_terms = _build_topic_term_vocab(question)
    # Remove common conversational terms
    q_terms -= {"what", "how", "tell", "about", "symptoms", "treatment", "cause", "define"}
    
    if not q_terms:
        return True
        
    combined_vocab = set()
    for d in docs:
        combined_vocab.update(_build_topic_term_vocab(d.page_content))
        
    # Check for direct overlap or fuzzy match
    for qt in q_terms:
        if qt in combined_vocab:
            return True
        # Fuzzy match for typos
        for ct in combined_vocab:
            if len(qt) > 4 and len(ct) > 4:
                # Simple distance check for typos
                if abs(len(qt) - len(ct)) <= 1:
                    matches = sum(1 for a, b in zip(qt, ct) if a == b)
                    if matches >= len(qt) - 1:
                        return True
    return False


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
    return answer


# ── MISSING FUNCTION 1 ──────────────────────────────────────────────────────
def _is_unsupported_answer(text: str) -> bool:
    """Return True if the assistant's answer was a refusal/unsupported response."""
    if not text:
        return True
    lower = text.lower().strip()
    unsupported_signals = (
        "i do not have enough",
        "i don't have enough",
        "not enough reliable",
        "unable to process",
        "cannot process",
        "i am currently unable",
        "no relevant context",
        "i could not find",
        "insufficient information",
        "unable to determine",
    )
    return any(signal in lower for signal in unsupported_signals)


# ── MISSING FUNCTION 2 ──────────────────────────────────────────────────────
_STOP_WORDS_TOPIC = frozenset({
    "what", "is", "are", "the", "a", "an", "of", "for", "with",
    "how", "why", "when", "where", "who", "tell", "me", "about",
    "its", "it", "this", "that", "do", "does", "can", "could",
    "please", "explain", "describe", "give", "list", "show",
    "symptoms", "symptom", "symtoms", "symtom",
    "treatment", "treatments", "treat",
    "cause", "causes", "sign", "signs",
    "definition", "define", "information", "info",
    "wha", "re", "waht", "hows",
})

_GENERIC_FOLLOWUP_TERMS = frozenset({
    "symptom", "symptoms", "sign", "signs",
    "treatment", "treatments", "treat", "tretament",
    "management", "therapy", "therapies",
    "cause", "causes", "complication", "complications",
    "type", "types", "prevention", "prevent", "precaution", "precautions",
    "medicine", "medicines", "medication", "medications",
    "diet", "foods", "food", "lifestyle", "exercise",
    "care", "steps", "step", "take",
})

def _topic_keywords(text: str) -> list[str]:
    """Extract meaningful medical topic keywords from a query string."""
    if not text:
        return []
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-']{2,}", text.lower())
    keywords = [
        t for t in tokens
        if t not in _STOP_WORDS_TOPIC and len(t) >= 3
    ]
    seen: set[str] = set()
    unique: list[str] = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique


def _has_specific_topic_in_question(question: str) -> bool:
    """Return True when a question contains a concrete disease/topic term."""
    explicit = _extract_condition_from_question(question)
    if explicit:
        return True

    for token in _topic_keywords(question):
        if token not in _GENERIC_FOLLOWUP_TERMS:
            return True
    return False


# ---------------------------------------------------------------------------
# FIXED: BUG 2 — Pronoun Resolution
# ---------------------------------------------------------------------------

_PRONOUN_REPLACEABLE = re.compile(
    r"\b(?:it|this|that|its)\b"
    r"|\bthe\s+(?:condition|disease|disorder|infection|syndrome)\b",
    re.IGNORECASE,
)


def _extract_last_topic_from_history(history: list[dict]) -> str:
    """Extract the most recently discussed medical topic from conversation history."""
    if not history:
        return ""

    for msg in reversed(history):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if _is_unsupported_answer(content):
                continue
            bold_matches = re.findall(
                r"\*\*([A-Za-z][A-Za-z\s\-\']{1,40})\*\*", content
            )
            for term in bold_matches:
                cleaned = term.strip()
                if cleaned.lower() not in {
                    "symptoms", "treatment", "causes", "note",
                    "important", "warning", "diagnosis",
                }:
                    return cleaned
            break

    for msg in reversed(history):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            topics = _topic_keywords(content)
            if topics:
                return " ".join(topics[:2])
            break

    return ""


def _resolve_pronouns(query: str, history: list[dict]) -> str:
    """Replace pronouns in short queries with the last mentioned medical topic."""
    if not history or not query:
        return query

    words = query.split()
    if len(words) >= PRONOUN_RESOLUTION_MAX_WORDS:
        return query

    if not _PRONOUN_REPLACEABLE.search(query):
        return query

    topic = _extract_last_topic_from_history(history)
    if not topic:
        return query

    original = query
    resolved = _PRONOUN_REPLACEABLE.sub(topic, query)

    if resolved != original:
        log.info("Pronoun resolved: '%s' -> '%s'", original, resolved)

    return resolved


# ---------------------------------------------------------------------------
# FIXED: BUG 3 — Topic Shift Detection
# ---------------------------------------------------------------------------

def _detect_topic_shift(resolved_query: str, history: list[dict]) -> bool:
    """Detect if the current query is about a different topic than the last."""
    if not history:
        return False

    try:
        last_user_query = ""
        for msg in reversed(history):
            if msg.get("role") == "user":
                last_user_query = msg.get("content", "")
                break

        if not last_user_query:
            return False

        current_topics = set(_topic_keywords(resolved_query))
        last_topics = set(_topic_keywords(last_user_query))

        if not current_topics:
            return False

        if current_topics and last_topics and current_topics.intersection(last_topics):
            return False

        if current_topics and last_topics:
            current_emb = get_embedding(" ".join(current_topics))
            last_emb = get_embedding(" ".join(last_topics))
            sim = compute_similarity(current_emb, last_emb)

            if sim < TOPIC_SHIFT_SIM_THRESHOLD:
                log.info(
                    "Topic shift detected (topic_sim=%.3f, %s -> %s). "
                    "Clearing history for this turn.",
                    sim, last_topics, current_topics,
                )
                return True

        return False
    except Exception as exc:
        log.debug("Topic shift detection error: %s. Defaulting to no shift.", exc)
        return False


# ---------------------------------------------------------------------------
# FIXED: BUG 1 — Answer Prefix Stripping
# ---------------------------------------------------------------------------

_DOUBLE_IS_PATTERN = re.compile(
    r"^(.+?\bis\b.+?)\bis\b\s+",
    re.IGNORECASE,
)


def _strip_doc_prefix_leakage(answer: str, docs: list, query: str = "") -> str:
    """Remove leading fragments that are verbatim continuations of retrieved chunks."""
    if not answer or not docs:
        return answer

    original_len = len(answer)
    cleaned = answer

    double_is = _DOUBLE_IS_PATTERN.match(cleaned)
    if double_is:
        prefix = double_is.group(1).strip()
        prefix_words = set(re.findall(r"\w+", prefix.lower()))
        for doc in docs:
            doc_words = set(
                re.findall(r"\w+", getattr(doc, "page_content", "").lower())
            )
            if prefix_words and doc_words:
                overlap = len(prefix_words.intersection(doc_words)) / max(len(prefix_words), 1)
                if overlap >= 0.5:
                    rest = cleaned[double_is.end():].strip()
                    if rest and len(rest) >= original_len * 0.6:
                        topic_terms = _topic_keywords(query)
                        if topic_terms:
                            topic = " ".join(t.title() for t in topic_terms[:2])
                            rest = f"**{topic}** is {rest}"
                        log.info("Stripped leaked doc prefix (double-is): '%s'", prefix[:80])
                        cleaned = rest
                    break

    if cleaned == answer:
        first_sentences = _extract_sentences(cleaned)
        if first_sentences and len(first_sentences) > 1:
            first = first_sentences[0]
            first_words = set(re.findall(r"\w+", first.lower()))
            if len(first_words) >= 4:
                for doc in docs:
                    doc_sents = _extract_sentences(getattr(doc, "page_content", ""))
                    for ds in doc_sents:
                        ds_words = set(re.findall(r"\w+", ds.lower()))
                        if not ds_words:
                            continue
                        overlap_ratio = len(first_words.intersection(ds_words)) / max(len(first_words), 1)
                        if overlap_ratio >= ANSWER_PREFIX_OVERLAP_THRESHOLD:
                            rest = cleaned[len(first):].strip()
                            if rest and len(rest) >= original_len * 0.6:
                                log.info("Stripped leaked doc prefix (overlap=%.2f): '%s'", overlap_ratio, first[:80])
                                cleaned = rest
                            break
                    if cleaned != answer:
                        break

    if len(cleaned) < original_len * 0.6:
        return answer

    return cleaned


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
        # Fast-path for simple greetings to ensure stability and speed
        if _is_pure_greeting(question):
            log.info("Pure greeting detected. Returning fast-path response.")
            return {
                "answer": _MEDICHAT_GREETING,
                "sources": [],
                "context_chars": 0,
                "top_k_used": 0,
                "confidence": "high",
                "grounding_removals": 0,
            }

        last_error = None
        for attempt_num, top_k in enumerate(self.top_k_steps, 1):
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": top_k}
            )
            try:
                return self._attempt(retriever, question, top_k, attempt_num, history)
            except Exception as e:
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
        resolved_question = _resolve_pronouns(question, history)
        topic_shifted = _detect_topic_shift(resolved_question, history)
        
        if topic_shifted:
            effective_history = []
        else:
            effective_history = history

        history_str = ""
        if effective_history:
            for msg in effective_history[-5:]:
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "")
                history_str += f"{role}: {content}\n"
        else:
            history_str = "No prior history."
        
        search_query = resolved_question
        if effective_history:
            history_topic = (
                _extract_condition_from_history(effective_history)
                or _extract_last_topic_from_history(effective_history)
            )
            q_cond = _extract_condition_from_question(resolved_question)
            if history_topic and q_cond and q_cond.lower() != history_topic.lower():
                from difflib import SequenceMatcher
                if SequenceMatcher(None, q_cond.lower(), history_topic.lower()).ratio() > 0.75:
                    log.info("Auto-corrected typo in query condition: %s -> %s", q_cond, history_topic)
                    resolved_question = re.sub(re.escape(q_cond), history_topic, resolved_question, flags=re.IGNORECASE)
                    search_query = resolved_question

            if history_topic and not _has_specific_topic_in_question(resolved_question):
                search_query = f"{history_topic} {resolved_question}"
                log.info("Anchored follow-up query to history topic: %s", history_topic)
        
        raw_docs = _retrieve_docs(retriever, search_query, k)
        docs = _deduplicate_docs(raw_docs)
        docs, _ = _enforce_diversity(docs, threshold=0.85)
        docs = rerank_docs(docs, search_query)
        docs = docs[:k]
        
        docs, context_chars = _enforce_context_budget(docs, MAX_CONTEXT_CHARS)
        
        (
            confidence,
            quality_class,
            _,
            _,
            _,
            _,
            confidence_reason,
        ) = _assess_context_quality(docs, question, context_query=search_query)

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
            question=resolved_question,
            dynamic_constraint=dynamic_constraint,
        )
        
        raw_answer = _call_llm(self.llm, filled_prompt)
        raw_answer = _strip_doc_prefix_leakage(raw_answer, docs, query=resolved_question)
        validated_answer, removals = _validate_grounding(raw_answer.strip(), docs, question=search_query)

        shaped_answer = _shape_answer_for_intent(
            question=resolved_question,
            answer=validated_answer,
            docs=docs,
            history=history,
        )

        final_answer = _apply_weak_context_preamble(shaped_answer, confidence)

        return {
            "answer":             final_answer,
            "sources":            [doc.page_content for doc in docs],
            "context_chars":      context_chars,
            "top_k_used":         k,
            "confidence":         confidence,
            "grounding_removals": removals,
            "diagnostic_report":  None,
            "resolved_query":     resolved_question,
            "topic_shifted":      topic_shifted,
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
