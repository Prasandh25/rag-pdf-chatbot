"""
Embeds chunks and stores them in a local Chroma collection.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from ingestion.config import get_embedding_function, PERSIST_DIR, COLLECTION_NAME


def embed_chunks(chunks: list[dict], source_filename: str = "unknown.pdf") -> None:
    """
    Generate embeddings for each chunk and store them in a persisted
    Chroma collection called "pdf_chunks".

    Args:
        chunks: List of dicts as returned by chunker.chunk_document(),
                each with 'text', 'source_page', 'chunk_id'.
        source_filename: Name of the PDF these chunks came from —
                          stored as metadata so multi-document
                          collections stay traceable.
    """
    if not chunks:
        print("[embedder] No chunks to embed — skipping.")
        return

    embedding_function = get_embedding_function()

    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "source_page": c["source_page"],
            "chunk_id": c["chunk_id"],
            "source_filename": source_filename,
        }
        for c in chunks
    ]
    # Chroma requires string ids; chunk_id (a uuid string) works directly.
    ids = [c["chunk_id"] for c in chunks]

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=PERSIST_DIR,
    )

    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    print(f"[embedder] Embedded and stored {len(chunks)} chunks "
          f"from '{source_filename}' into collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    # quick manual smoke test — run: python ingestion/embedder.py
    from ingestion.pdf_loader import load_pdf
    from ingestion.chunker import chunk_document

    test_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "sample_pdfs", "research_paper.pdf"
    )

    pages = load_pdf(test_path)
    chunks = chunk_document(pages)
    embed_chunks(chunks, source_filename="research_paper.pdf")