"""
MediChat Knowledge Bank: This module handles building and loading our 
searchable medical database.

We use FAISS to store our clinical data as mathematical vectors, which 
allows the chatbot to quickly find relevant medical literature for any 
given question.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from src.config import VECTOR_STORE_PATH, CHUNK_SIZE, CHUNK_OVERLAP, SAFE_DESERIALIZATION
from src.logger import get_logger
from src.embedder import get_embedding_model

log = get_logger(__name__)


def build_vector_store(documents: list[str]) -> FAISS:
    """
    Chunk `documents`, embed them, and save the FAISS index to disk.

    Args:
        documents: Raw text documents (one per clinical Q&A pair).

    Returns:
        The in-memory FAISS vector store.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.create_documents(documents)
    log.info("Split %d documents → %d chunks.", len(documents), len(chunks))

    embeddings = get_embedding_model()
    log.info("Building FAISS index…")
    vector_store = FAISS.from_documents(chunks, embeddings)

    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTOR_STORE_PATH))
    log.info("FAISS index saved to: %s", VECTOR_STORE_PATH)

    return vector_store


def load_vector_store() -> FAISS:
    """
    Load a previously saved FAISS index from disk.

    Raises:
        FileNotFoundError: If the index has not been built yet.
    """
    if not VECTOR_STORE_PATH.exists():
        raise FileNotFoundError(
            f"No vector store found at '{VECTOR_STORE_PATH}'. "
            "Run the ingestion pipeline first."
        )

    embeddings = get_embedding_model()
    log.info("Loading FAISS index from: %s", VECTOR_STORE_PATH)

    # SAFE_DESERIALIZATION=false is required only if you built the index with
    # a LangChain version that pickles metadata.  Default is True (safe).
    # If you see a deserialization error, verify the index came from this
    # project's own ingestion step, then set SAFE_DESERIALIZATION=false.
    allow_pickle = not SAFE_DESERIALIZATION
    if allow_pickle:
        log.warning(
            "SAFE_DESERIALIZATION is disabled.  Ensure the FAISS index was "
            "produced by this project's ingestion pipeline and has not been "
            "tampered with before loading."
        )

    return FAISS.load_local(
        str(VECTOR_STORE_PATH),
        embeddings,
        allow_dangerous_deserialization=allow_pickle,
    )
