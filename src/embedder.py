"""
embedder.py – Loads and provides the embedding model.

Responsibilities:
  - Initialize the sentence-transformer model for embeddings
"""

from functools import lru_cache
import hashlib
import re

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL
from src.logger import get_logger

log = get_logger(__name__)


class _FallbackEmbeddings:
    """Deterministic local embedding fallback when model download is unavailable."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"[a-zA-Z0-9\-]+", (text or "").lower())
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[idx] += sign

        norm = sum(v * v for v in vec) ** 0.5
        if norm == 0:
            return vec
        return [v / norm for v in vec]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(t) for t in texts]

    def __call__(self, text: str) -> list[float]:
        # Compatibility for vector stores that still treat embedding functions as callables.
        return self.embed_query(text)


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load and cache the sentence-transformer embedding model.
    
    Returns:
        Embedding model with embed_query and embed_documents methods.
    """
    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    try:
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Embedding model unavailable (%s). Falling back to deterministic local embeddings.",
            exc,
        )
        return _FallbackEmbeddings()
