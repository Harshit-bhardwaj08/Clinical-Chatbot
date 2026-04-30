"""
MediChat API Server: This is the FastAPI backend that powers our RAG pipeline.

It mainly handles incoming queries via the /query endpoint, ensuring 
inputs are validated and rate-limited to keep the system stable and secure.

To start the server:
    uvicorn app.api_server:app --host 0.0.0.0 --port 8000 --reload
"""

import time
import collections
import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.vector_store import load_vector_store
from src.rag_chain import build_rag_chain, query
from src.config import MAX_QUERY_LENGTH, RATE_LIMIT_PER_MIN
from src.logger import get_logger

log = get_logger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Clinical RAG API",
    description="Backend API for the Clinical RAG Chatbot",
    version="1.0.0",
)

# Restrict CORS to localhost so the API is not callable from arbitrary origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ── Basic Rate Limiting ──
# We use a simple thread-safe deque to track request timestamps per IP.
# This keeps the API responsive without needing an external database.

_rate_store: dict[str, collections.deque] = {}
_rate_lock = threading.Lock()


def _is_rate_limited(ip: str) -> bool:
    """Return True if the given IP has exceeded RATE_LIMIT_PER_MIN requests/min."""
    now = time.monotonic()
    window = 60.0  # sliding 60-second window

    with _rate_lock:
        if ip not in _rate_store:
            _rate_store[ip] = collections.deque()

        timestamps = _rate_store[ip]

        # Drop timestamps older than the window
        while timestamps and now - timestamps[0] > window:
            timestamps.popleft()

        if len(timestamps) >= RATE_LIMIT_PER_MIN:
            return True

        timestamps.append(now)
        return False


# ── Request / Response models ────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    history: list[dict] = []


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: str = "n/a"
    grounding_removals: int = 0


# ── Lazy-loading the RAG Chain ──
# We build the heavy RAG components only when the first request comes in.
# This makes the initial server startup almost instantaneous.
_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        log.info("Loading vector store and building RAG chain (first request)...")
        vector_store = load_vector_store()
        _chain = build_rag_chain(vector_store)
        log.info("RAG chain ready.")
    return _chain


def _warm_chain_background() -> None:
    """Warm retrieval + LLM chain on startup so first query is faster."""
    try:
        _get_chain()
    except Exception as exc:  # noqa: BLE001
        log.warning("Background warmup failed: %s", exc)


@app.on_event("startup")
def _startup_warmup() -> None:
    """Start non-blocking warmup to reduce first-request latency."""
    threading.Thread(target=_warm_chain_background, daemon=True).start()


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def handle_query(req: QueryRequest, request: Request):
    # ── Rate limiting ──────────────────────────────────────────────────────
    client_ip = request.client.host if request.client else "unknown"
    if _is_rate_limited(client_ip):
        log.warning("Rate limit exceeded for IP: %s", client_ip)
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a moment before trying again.",
        )

    # ── Input validation ───────────────────────────────────────────────────
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    if len(req.question) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Query is too long ({len(req.question)} characters). "
                f"Please keep it under {MAX_QUERY_LENGTH} characters."
            ),
        )

    # ── Pipeline ───────────────────────────────────────────────────────────
    try:
        chain = _get_chain()
        result = query(chain, req.question, history=req.history)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result.get("confidence", "n/a"),
            grounding_removals=result.get("grounding_removals", 0),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        log.error("Query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing error.")
