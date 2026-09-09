"""
Loads the persisted Chroma vector store so it can be reused across
runs without re-embedding already-indexed documents.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from ingestion.config import get_embedding_function, PERSIST_DIR, COLLECTION_NAME


def get_vectorstore() -> Chroma:
    """
    Load the persisted 'pdf_chunks' Chroma collection from disk.

    Returns:
        A Chroma vectorstore instance backed by the on-disk collection.
        If no collection exists yet at PERSIST_DIR, Chroma creates an
        empty one — callers should check chunk count before assuming
        data is present.
    """
    embedding_function = get_embedding_function()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=PERSIST_DIR,
    )
    return vectorstore


if __name__ == "__main__":
    # quick manual smoke test — run: python retrieval/vector_store.py
    vs = get_vectorstore()
    count = vs._collection.count()
    print(f"[vector_store] Loaded collection '{COLLECTION_NAME}' with {count} chunks.")

    if count > 0:
        results = vs.similarity_search("productivity", k=2)
        for r in results:
            print(f"\n--- page {r.metadata.get('source_page')} "
                  f"({r.metadata.get('source_filename')}) ---")
            print(r.page_content[:200])