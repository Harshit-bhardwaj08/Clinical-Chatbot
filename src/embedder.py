"""
embedder.py – Loads and provides the embedding model.

Responsibilities:
  - Initialize the sentence-transformer model for embeddings
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL
from src.logger import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load and cache the sentence-transformer embedding model.
    
    Returns:
        HuggingFaceEmbeddings: The loaded embedding model instance.
    """
    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

