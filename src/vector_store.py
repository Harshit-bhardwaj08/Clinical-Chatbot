"""
MediChat Knowledge Bank: build and load the searchable medical database.
"""

import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from src.config import VECTOR_STORE_PATH, CHUNK_SIZE, CHUNK_OVERLAP, SAFE_DESERIALIZATION
from src.logger import get_logger
from src.embedder import get_embedding_model

log = get_logger(__name__)

_PROCESSED_DATA_PATH = VECTOR_STORE_PATH.parent / "processed.json"
_MIN_HEALTHY_INDEX_SIZE = 100


def build_vector_store(documents: list[str], save: bool = True) -> FAISS:
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
    log.info("Split %d documents -> %d chunks.", len(documents), len(chunks))

    embeddings = get_embedding_model()
    log.info("Building FAISS index…")
    vector_store = FAISS.from_documents(chunks, embeddings)

    if save:
        VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(VECTOR_STORE_PATH))
        log.info("FAISS index saved to: %s", VECTOR_STORE_PATH)

    return vector_store


def _vector_store_size(vector_store: FAISS) -> int:
    """Return the number of vectors in a loaded FAISS store, if available."""
    index = getattr(vector_store, "index", None)
    return int(getattr(index, "ntotal", 0) or 0)


def _load_processed_documents() -> list[str]:
    """Load already-preprocessed local documents for offline index recovery."""
    if not _PROCESSED_DATA_PATH.exists():
        return []

    records = json.loads(_PROCESSED_DATA_PATH.read_text(encoding="utf-8"))
    docs: list[str] = []
    for record in records:
        if isinstance(record, dict):
            text = str(record.get("text", "")).strip()
        else:
            text = str(record).strip()
        if text:
            docs.append(text)
    return docs


def _rebuild_from_processed_if_available() -> FAISS | None:
    """Rebuild FAISS from data/processed.json when the saved index is tiny."""
    docs = _load_processed_documents()
    if not docs:
        return None

    log.warning(
        "Vector store looks incomplete; rebuilding from %s (%d documents).",
        _PROCESSED_DATA_PATH,
        len(docs),
    )
    return build_vector_store(docs)


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

    vector_store = FAISS.load_local(
        str(VECTOR_STORE_PATH),
        embeddings,
        allow_dangerous_deserialization=allow_pickle,
    )

    size = _vector_store_size(vector_store)
    if size and size < _MIN_HEALTHY_INDEX_SIZE:
        rebuilt = _rebuild_from_processed_if_available()
        if rebuilt is not None:
            return rebuilt

    return vector_store
