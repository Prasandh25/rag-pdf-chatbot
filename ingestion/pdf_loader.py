"""
PDF loading utilities. Extracts text from a PDF page by page.
"""

from pypdf import PdfReader
from pypdf.errors import PdfReadError


def load_pdf(path: str) -> list[dict]:
    """
    Extract text from a PDF, page by page.

    Args:
        path: Path to the PDF file.

    Returns:
        A list of dicts, one per page, each with:
            - page_number (int): 1-indexed page number
            - text (str): extracted text, or "" if extraction failed
                          (e.g. scanned/image-only page with no text layer)

    Raises:
        FileNotFoundError: if the path doesn't exist.
        PdfReadError: if the file isn't a valid/readable PDF at all.
    """
    reader = PdfReader(path)
    pages: list[dict] = []

    for i, page in enumerate(reader.pages):
        page_number = i + 1
        try:
            text = page.extract_text() or ""
        except Exception as e:
            # A single bad page shouldn't crash the whole load —
            # log and move on with an empty string for that page.
            print(f"[pdf_loader] Warning: failed to extract text from "
                  f"page {page_number} of '{path}': {e}")
            text = ""

        pages.append({
            "page_number": page_number,
            "text": text.strip(),
        })

    return pages


if __name__ == "__main__":
    # quick manual smoke test — run: python ingestion/pdf_loader.py
    import sys
    import os

    test_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "data", "sample_pdfs", "research_paper.pdf"
    )

    result = load_pdf(test_path)
    print(f"Loaded {len(result)} pages from {test_path}\n")
    for p in result[:2]:
        preview = p["text"][:200] + ("..." if len(p["text"]) > 200 else "")
        print(f"--- Page {p['page_number']} ---")
        print(preview or "(no extractable text)")
        print()