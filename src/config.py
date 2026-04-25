"""
MediChat Configuration: This is where we load all our app settings.

Everything is driven by the .env file. Instead of hardcoding values 
here, we pull them from environment variables to make the app easy 
to deploy in different environments.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Locate the project root (two levels up from this file: src/config.py → root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root
load_dotenv(PROJECT_ROOT / ".env")


# ── The Models (Ollama & Embeddings) ──
# We use Ollama for our LLM and a local SentenceTransformer for embeddings.
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Data Storage (Vector Store & Raw Data) ──
VECTOR_STORE_PATH: Path = PROJECT_ROOT / os.getenv(
    "VECTOR_STORE_PATH", "data/vector_store"
)
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_NAME: str   = os.getenv("DATASET_NAME",   "medalpaca/medical_meadow_medical_flashcards")
DATASET_SPLIT: str  = os.getenv("DATASET_SPLIT",  "train")
DATASET_SUBSET: str = os.getenv("DATASET_SUBSET", "default")

# Path where the raw downloaded dataset is saved for offline inspection
RAW_DATA_PATH: Path = PROJECT_ROOT / os.getenv("RAW_DATA_PATH", "data/raw.json")

# ── Data Loader retry behaviour ────────────────────────────────────────────────
MAX_RETRIES: int          = int(float(os.getenv("MAX_RETRIES",          "3")))
RETRY_DELAY_SECONDS: float = float(os.getenv("RETRY_DELAY_SECONDS", "5"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "15"))

# ── Second-stage Reranker ─────────────────────────────────────────────────────
# Minimum composite score (keyword overlap + similarity blend) a chunk must
# reach to be passed to the LLM.  Range: [0.0, 1.0].
# 0.0  = no filtering (pure FAISS order, reranker only re-sorts).
# 0.15 = light filter (keeps chunks with at least ~1 keyword hit).
# 0.30 = strict filter (recommended for precise clinical queries).
RERANK_MIN_SCORE: float = float(os.getenv("RERANK_MIN_SCORE", "0.15"))

# Maximum number of chunks to pass to the LLM after reranking.
# Acts as an additional cap on top of MAX_CONTEXT_CHARS.
# 0 = no cap (all chunks that pass the score threshold are forwarded).
RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "4"))

# ── RAG chain robustness ───────────────────────────────────────────────────────
# Maximum total characters of retrieved context sent to the LLM.
# Prevents "llama runner process has terminated" on context-window overflow.
# Rule of thumb: Ollama llama3 8B has ~8k token window ≈ 32 000 chars.
# Use a conservative 6 000 to leave room for the system header + question.
# Set to 0 to disable budget enforcement (not recommended).
MAX_CONTEXT_CHARS: int = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))

# Fallback TOP_K values used when the LLM fails with the primary TOP_K.
# The chain retries in order: TOP_K_RESULTS → MEDIUM → MIN
FALLBACK_TOP_K_MEDIUM: int = int(os.getenv("FALLBACK_TOP_K_MEDIUM", "4"))
FALLBACK_TOP_K_MIN: int    = int(os.getenv("FALLBACK_TOP_K_MIN",    "2"))

# Seconds before an Ollama request is considered timed out.
LLM_TIMEOUT_SECONDS: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))


# ── App UI ────────────────────────────────────────────────────────────────────
APP_TITLE: str = os.getenv("APP_TITLE", "Clinical RAG Chatbot")
APP_ICON: str = os.getenv("APP_ICON", "🏥")
API_URL: str = os.getenv("API_URL", "http://localhost:8000/query")

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ── Security & Safety ─────────────────────────────────────────────────────────
# Maximum characters accepted in a single user query.
# Queries longer than this are rejected at the API layer to prevent
# prompt injection and context-window overflow attacks.
MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "500"))

# When True, FAISS deserialization is performed without allowing arbitrary
# pickle execution (safer for shared/demo environments).
# Set to False only if you built the index yourself and trust the source.
SAFE_DESERIALIZATION: bool = os.getenv("SAFE_DESERIALIZATION", "true").lower() == "true"

# ── Rate Limiting ─────────────────────────────────────────────────────────────
# Maximum API requests per minute allowed from a single IP address.
# Enforced in-memory by the FastAPI server; no external dependency required.
RATE_LIMIT_PER_MIN: int = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))

# ── Preprocessing ─────────────────────────────────────────────────────────────
# Convert all text to lowercase before embedding?
# Improves recall for simple queries; may hurt precision for drug names.
PREPROCESSING_LOWERCASE: bool = os.getenv("PREPROCESSING_LOWERCASE", "false").lower() == "true"

# Maximum word count per document (0 = no limit).
# Documents above this limit are hard-truncated to protect the embedding model.
PREPROCESSING_MAX_WORDS: int = int(os.getenv("PREPROCESSING_MAX_WORDS", "0"))

# Minimum word count per document.
# Near-empty documents (header only, no body) are dropped.
PREPROCESSING_MIN_WORDS: int = int(os.getenv("PREPROCESSING_MIN_WORDS", "10"))


# ── Debug / Diagnostics ───────────────────────────────────────────────────────
# Master switch.  Set DEBUG_MODE=true in .env to activate verbose pipeline
# diagnostics.  ALWAYS leave this False in shared or demo environments.
DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
