"""
Retriever: searches the vector store for the most relevant chunks
to a given query, with a tunable relevance threshold to filter out
low-confidence matches.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.vector_store import get_vectorstore

# Chroma's default similarity_search_with_score returns a DISTANCE
# (lower = more similar) for the default L2 metric, not a similarity
# score. Threshold is tuned for that — see notes below before changing it.
DEFAULT_DISTANCE_THRESHOLD = 1.0


def retrieve(
    query: str,
    k: int = 4,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a query, filtering out
    low-relevance results.

    Args:
        query: The user's question.
        k: Number of top candidates to fetch before filtering.
        distance_threshold: Max allowed distance (lower = more similar).
                             Results with distance above this are dropped.
                             Tune this against your own data — see
                             the __main__ block for how to inspect
                             real distance values.

    Returns:
        A list of dicts, each with:
            - text (str): the chunk's content
            - source_page (int): which page it came from
            - source_filename (str): which document it came from
            - chunk_id (str)
            - score (float): the raw distance value (lower = better match)
    """
    vectorstore = get_vectorstore()

    # similarity_search_with_score returns (Document, distance) tuples
    results = vectorstore.similarity_search_with_score(query, k=k)

    filtered = []
    for doc, distance in results:
        if distance > distance_threshold:
            continue  # too dissimilar — likely irrelevant, drop it
        filtered.append({
            "text": doc.page_content,
            "source_page": doc.metadata.get("source_page"),
            "source_filename": doc.metadata.get("source_filename"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "score": distance,
        })

    return filtered


if __name__ == "__main__":
    # quick manual smoke test — run: python retrieval/retriever.py
    # Try a question you KNOW the answer to, and a nonsense one, to
    # calibrate distance_threshold for your own embedding model/data.

    test_queries = [
        "What productivity increase did asynchronous teams show?",  # answerable
        "What is the home office setup stipend?",                   # answerable (diff doc)
        "What is the best recipe for chocolate cake?",               # off-topic / nonsense
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"QUERY: {q}")
        print("=" * 70)

        # Print RAW results first (no filtering) so you can see actual
        # distance values and calibrate DEFAULT_DISTANCE_THRESHOLD yourself.
        vs = get_vectorstore()
        raw_results = vs.similarity_search_with_score(q, k=4)
        print("\n-- raw distances (unfiltered) --")
        for doc, distance in raw_results:
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"  distance={distance:.4f} | page={doc.metadata.get('source_page')} "
                  f"| {preview}...")

        # Now show what the filter actually keeps
        filtered = retrieve(q, k=4)
        print(f"\n-- after filtering (threshold={DEFAULT_DISTANCE_THRESHOLD}) --")
        if not filtered:
            print("  (no results passed the relevance filter)")
        for r in filtered:
            preview = r["text"][:80].replace("\n", " ")
            print(f"  score={r['score']:.4f} | page={r['source_page']} | {preview}...")