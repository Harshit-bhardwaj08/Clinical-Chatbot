"""
retriever.py – Wraps the FAISS vector store as a LangChain retriever.

Provides a single entry-point function that takes a user query and
returns the top-k most relevant document chunks.
"""

from langchain_community.vectorstores import FAISS

from src.config import TOP_K_RESULTS
from src.logger import get_logger

log = get_logger(__name__)


def get_retriever(vector_store: FAISS):
    """
    Create a similarity-search retriever from the FAISS vector store.

    Args:
        vector_store: A loaded or freshly-built FAISS instance.

    Returns:
        A LangChain retriever configured for top-k similarity search.
    """
    log.info("Creating retriever (top_k=%d).", TOP_K_RESULTS)
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )
