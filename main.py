"""
main.py – Entry point for the Clinical RAG Chatbot pipeline.

Usage:
    python main.py --check      # Verify environment setup (default)
    python main.py --ingest     # Download dataset and build FAISS index
    python main.py --query "What are the symptoms of diabetes?"
"""

import argparse
import sys

from src.logger import get_logger
from src.config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    VECTOR_STORE_PATH,
    DATASET_NAME,
    DATASET_SUBSET,
    DATASET_SPLIT,
    TOP_K_RESULTS,
    DEBUG_MODE,
)

log = get_logger(__name__)


def check_environment():
    """Print the active configuration and confirm all imports are healthy."""
    log.info("── Environment Check ────────────────────────────────")
    log.info("  LLM model       : %s", OLLAMA_MODEL)
    log.info("  Ollama URL      : %s", OLLAMA_BASE_URL)
    log.info("  Embedding model : %s", EMBEDDING_MODEL)
    log.info("  Vector store    : %s", VECTOR_STORE_PATH)
    log.info("  Dataset         : %s / %s (%s)", DATASET_NAME, DATASET_SUBSET, DATASET_SPLIT)
    log.info("  Top-k results   : %d", TOP_K_RESULTS)
    log.info("─────────────────────────────────────────────────────")

    # Confirm heavy imports load without error
    log.info("Verifying imports…")
    try:
        import langchain          # noqa: F401
        import faiss              # noqa: F401
        import sentence_transformers  # noqa: F401
        import streamlit          # noqa: F401
        import ollama             # noqa: F401
        import datasets           # noqa: F401
        import sklearn            # noqa: F401
        log.info("All imports OK ✓")
    except ImportError as exc:
        log.error("Import failed: %s", exc)
        log.error("Run:  pip install -r requirements.txt")
        sys.exit(1)

    log.info("Environment check passed. You're good to go!")


def run_ingestion():
    """Download the dataset, preprocess documents, and build the FAISS vector store."""
    from src.data_loader import load_clinical_documents
    from src.preprocessing import preprocess_documents
    from src.vector_store import build_vector_store

    documents = load_clinical_documents()
    documents = preprocess_documents(documents)
    build_vector_store(documents)
    log.info("Ingestion complete. Vector store saved to: %s", VECTOR_STORE_PATH)


def run_query(question: str):
    """Run a single question through the RAG pipeline and print the answer."""
    from src.vector_store import load_vector_store
    from src.rag_chain import build_rag_chain, query

    vector_store = load_vector_store()
    chain = build_rag_chain(vector_store)

    result = query(chain, question)

    confidence = result.get("confidence", "n/a").upper()
    removals   = result.get("grounding_removals", 0)
    confidence_icon = {"HIGH": "[HIGH]", "MEDIUM": "[MED]", "LOW": "[LOW]", "N/A": "[N/A]"}.get(confidence, "[?]")
    removal_note = (
        f"  [REMOVED] {removals} hallucinated sentence(s) removed"
        if removals else "  [OK] No hallucinations detected"
    )
    debug_indicator = "  [DEBUG MODE ACTIVE]" if DEBUG_MODE else ""

    print("\n" + "=" * 60)
    print(f"Question   : {question}")
    print(f"Confidence : {confidence_icon} {confidence}")
    print(f"Grounding  :{removal_note}")
    if debug_indicator:
        print(debug_indicator)
    print("=" * 60)
    print(f"Answer:\n{result['answer']}")
    print("-" * 60)
    print(f"Sources  : {len(result['sources'])} chunks retrieved")
    for i, src in enumerate(result["sources"], 1):
        print(f"  [{i}] {src[:120]}...")

    # When DEBUG_MODE is on, the full report is already printed by diagnostics.py.
    # Print a brief hint here for quick CLI visibility.
    diag = result.get("diagnostic_report")
    if diag:
        print("-" * 60)
        print("  [DIAGNOSTIC SUMMARY]")
        print(f"     Root cause : {diag['root_cause'][:80]}...")
        print(f"     Suggested  : {diag['recommended_fix'][:80]}...")
        print("     (Full report printed above by the diagnostic engine)")

    print("=" * 60 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Clinical RAG Chatbot – pipeline entry point"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        action="store_true",
        default=True,
        help="Verify environment setup (default action)",
    )
    group.add_argument(
        "--ingest",
        action="store_true",
        help="Download dataset and build the FAISS vector store",
    )
    group.add_argument(
        "--query",
        metavar="QUESTION",
        type=str,
        help="Ask a single question via the RAG pipeline",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.ingest:
        run_ingestion()
    elif args.query:
        run_query(args.query)
    else:
        check_environment()
