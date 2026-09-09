"""
CLI entry point for the RAG PDF chatbot.

On startup:
  1. Scans data/sample_pdfs/ for PDFs not yet indexed in the vector store.
  2. Loads, chunks, and embeds any new ones.
  3. Enters an interactive Q&A loop, printing answers with cited source pages.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_document
from ingestion.embedder import embed_chunks
from retrieval.vector_store import get_vectorstore
from retrieval.retriever import retrieve
from generation.answer_generator import generate_answer

SAMPLE_PDFS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_pdfs")


def get_already_indexed_filenames() -> set[str]:
    """
    Inspect the vector store's metadata to find which source filenames
    have already been embedded, so we don't re-index (and duplicate
    cost/time on) PDFs we've already processed.
    """
    vectorstore = get_vectorstore()
    collection = vectorstore._collection
    count = collection.count()

    if count == 0:
        return set()

    # Pull all metadata to see which source_filenames are present.
    all_data = collection.get(include=["metadatas"])
    filenames = {
        meta["source_filename"]
        for meta in all_data["metadatas"]
        if meta and "source_filename" in meta
    }
    return filenames


def ingest_new_pdfs() -> None:
    """
    Find PDFs in data/sample_pdfs/ that aren't already indexed, and
    embed them. Prints a summary of what was (or wasn't) newly indexed.
    """
    if not os.path.isdir(SAMPLE_PDFS_DIR):
        print(f"[main] No sample_pdfs directory found at {SAMPLE_PDFS_DIR}")
        return

    all_pdfs = [f for f in os.listdir(SAMPLE_PDFS_DIR) if f.lower().endswith(".pdf")]
    if not all_pdfs:
        print(f"[main] No PDFs found in {SAMPLE_PDFS_DIR}")
        return

    already_indexed = get_already_indexed_filenames()
    new_pdfs = [f for f in all_pdfs if f not in already_indexed]

    if not new_pdfs:
        print(f"[main] All {len(all_pdfs)} PDF(s) already indexed. Nothing to do.")
        return

    print(f"[main] Found {len(new_pdfs)} new PDF(s) to index: {new_pdfs}")

    for filename in new_pdfs:
        path = os.path.join(SAMPLE_PDFS_DIR, filename)
        print(f"[main] Ingesting '{filename}'...")

        pages = load_pdf(path)
        chunks = chunk_document(pages)

        if not chunks:
            print(f"[main] Warning: '{filename}' produced no chunks (empty or unreadable). Skipping.")
            continue

        embed_chunks(chunks, source_filename=filename)

    print("[main] Ingestion complete.\n")


def format_sources(pages: list[int]) -> str:
    if not pages:
        return "none"
    return ", ".join(f"page {p}" for p in sorted(set(pages)))


def run_qa_loop() -> None:
    """
    Interactive loop: user types a question, gets back a generated
    answer with cited source pages. Type 'exit' or 'quit' to stop.
    """
    print("=" * 70)
    print("RAG PDF Chatbot — ask a question about the indexed documents.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 70)

    while True:
        try:
            query = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[main] Exiting.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("[main] Exiting.")
            break

        chunks = retrieve(query, k=4)
        result = generate_answer(query, chunks)

        print(f"\nAnswer: {result['answer']}")
        print(f"Sources: {format_sources(result['sources'])}")


def main():
    ingest_new_pdfs()
    run_qa_loop()


if __name__ == "__main__":
    main()