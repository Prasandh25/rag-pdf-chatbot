"""
Answer generation: builds a grounded prompt from retrieved chunks,
calls the LLM, and returns a structured answer with cited sources.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

MODEL_NAME = "gemini-3.6-flash"
FALLBACK_ANSWER = "I don't know based on the provided document."


def _build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """Builds the grounded prompt, with each chunk labeled by page for citation."""
    context_blocks = []
    for chunk in retrieved_chunks:
        context_blocks.append(
            f"[Page {chunk['source_page']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    return f"""You are a document Q&A assistant. Answer the question using ONLY the
context below. Do not use any outside knowledge.

If the context does not contain enough information to answer the question,
respond with exactly: "{FALLBACK_ANSWER}"
Do not guess or make up an answer that isn't supported by the context.

Context:
{context}

Question: {query}

Respond ONLY with valid JSON in this exact format, no markdown fences, no other text:
{{
  "answer": "<your answer, or the fallback line above>",
  "pages_used": [<list of page numbers from the context that you actually used to answer, empty list if none>]
}}"""

def _extract_text(content) -> str:
    """
    Normalize LangChain message content into a plain string.
    Some models (like gemini-3.6-flash) return content as a list of
    blocks (e.g. [{"type": "text", "text": "..."}]) instead of a
    plain string — this handles both shapes.
    """
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # most common shape: {"type": "text", "text": "..."}
                text_val = block.get("text", "")
                parts.append(text_val)
        return "".join(parts).strip()

    return str(content).strip()
def generate_answer(query: str, retrieved_chunks: list[dict]) -> dict:
    """
    Generate a grounded answer from retrieved chunks.

    Args:
        query: The user's question.
        retrieved_chunks: List of dicts from retriever.retrieve(), each with
                           'text', 'source_page', 'source_filename', etc.

    Returns:
        A dict with:
            - answer (str): the generated answer, or the "I don't know" fallback
            - sources (list[int]): source_page numbers actually used in the answer
    """
    # No chunks retrieved at all — don't even call the LLM, just say so.
    if not retrieved_chunks:
        return {"answer": FALLBACK_ANSWER, "sources": []}

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)
    prompt = _build_prompt(query, retrieved_chunks)

    response = llm.invoke(prompt)
    raw_text = _extract_text(response.content)

    # Model sometimes wraps JSON in markdown fences despite instructions —
    # strip those defensively before parsing.
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()

    try:
        parsed = json.loads(raw_text)
        answer = parsed.get("answer", FALLBACK_ANSWER)
        pages_used = parsed.get("pages_used", [])
        # Guard against the model returning junk types
        if not isinstance(pages_used, list):
            pages_used = []
        sources = [p for p in pages_used if isinstance(p, int)]
    except (json.JSONDecodeError, AttributeError):
        # If the model didn't return valid JSON, fall back to treating
        # the raw text as the answer with no verified sources — better
        # than crashing, but worth logging so you notice if it happens often.
        print(f"[answer_generator] Warning: could not parse JSON response:\n{raw_text}")
        answer = raw_text
        sources = []

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    # quick manual smoke test — run: python generation/answer_generator.py
    from retrieval.retriever import retrieve

    test_queries = [
        "What productivity increase did asynchronous teams show?",  # answerable
        "What is the home office setup stipend?",                    # answerable
        "What is the best recipe for chocolate cake?",                # NOT in context
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"QUERY: {q}")
        print("=" * 70)

        chunks = retrieve(q, k=4)
        result = generate_answer(q, chunks)

        print(f"ANSWER: {result['answer']}")
        print(f"SOURCES: {result['sources']}")