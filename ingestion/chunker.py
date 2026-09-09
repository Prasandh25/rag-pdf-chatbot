"""
Chunking utilities. Splits page-level text into overlapping chunks
suitable for embedding, while preserving source page metadata.
"""

import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_document(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    Split page-level text into chunks for embedding.

    Args:
        pages: List of dicts as returned by pdf_loader.load_pdf(),
               each with 'page_number' and 'text'.
        chunk_size: Max characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks,
                        so context isn't lost at chunk boundaries.

    Returns:
        A list of dicts, one per chunk, each with:
            - text (str): the chunk's text content
            - source_page (int): which page this chunk came from
            - chunk_id (str): unique id for this chunk
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try to split on paragraph breaks first, then sentences,
        # then words — falls back to hard character splits only
        # as a last resort. This is what keeps chunks from cutting
        # mid-sentence when there's a cleaner boundary available.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []

    for page in pages:
        page_number = page["page_number"]
        text = page["text"]

        if not text:
            # Skip pages with no extractable text (e.g. scanned images) —
            # nothing to chunk, and an empty chunk would just be noise
            # in the vector store.
            continue

        page_chunks = splitter.split_text(text)

        for chunk_text in page_chunks:
            chunks.append({
                "text": chunk_text,
                "source_page": page_number,
                "chunk_id": str(uuid.uuid4()),
            })

    return chunks


if __name__ == "__main__":
    # quick manual smoke test — run: python ingestion/chunker.py
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ingestion.pdf_loader import load_pdf

    test_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "sample_pdfs", "research_paper.pdf"
    )

    pages = load_pdf(test_path)
    chunks = chunk_document(pages)

    print(f"Loaded {len(pages)} pages -> produced {len(chunks)} chunks\n")
    for c in chunks[:5]:
        print(f"--- chunk_id={c['chunk_id'][:8]}... | source_page={c['source_page']} ---")
        print(c["text"])
        print(f"(length: {len(c['text'])} chars)\n")