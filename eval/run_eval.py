"""
Evaluation harness for the RAG pipeline.

Reports two metrics against eval/ground_truth.json:
  - Retrieval precision: did the retriever pull back a chunk from the
    correct source page/document for each question?
  - Answer correctness: does the generated answer contain the expected
    key facts (keyword-based check against expected_keywords)?

This is a manual scorer rather than an LLM-as-judge or ragas pipeline —
deliberately simple and fully transparent: every score traces to a
concrete, inspectable rule, with no extra API calls or dependency
surface beyond what Steps 5/6 already use.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.retriever import retrieve
from generation.answer_generator import generate_answer

GROUND_TRUTH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth.json")


def load_ground_truth() -> list[dict]:
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def score_retrieval(retrieved_chunks: list[dict], expected_filename: str | None, expected_page: int | None) -> bool:
    """
    Retrieval precision for a single question: True if at least one
    retrieved chunk matches BOTH the expected source file and page.

    For the deliberately-unanswerable question (expected_filename=None),
    this instead checks that NOTHING relevant was retrieved — an empty
    or irrelevant retrieval is the "correct" retrieval behavior there.
    """
    if expected_filename is None:
        # Unanswerable question: correct retrieval = no chunks passed
        # the relevance filter (or none from a real document context).
        return len(retrieved_chunks) == 0

    for chunk in retrieved_chunks:
        if (chunk.get("source_filename") == expected_filename
                and chunk.get("source_page") == expected_page):
            return True
    return False


def score_answer(generated_answer: str, expected_keywords: list[str]) -> tuple[bool, float]:
    """
    Answer correctness for a single question: fraction of expected
    keywords found (case-insensitive) in the generated answer.
    Considered "correct" if ALL expected keywords are present —
    a strict but simple and explainable bar.
    """
    answer_lower = generated_answer.lower()
    matched = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    total = len(expected_keywords)
    match_ratio = matched / total if total > 0 else 0.0
    is_correct = matched == total
    return is_correct, match_ratio


def run_eval():
    ground_truth = load_ground_truth()
    results = []

    for item in ground_truth:
        query = item["question"]
        retrieved_chunks = retrieve(query, k=4)
        generation_result = generate_answer(query, retrieved_chunks)
        generated_answer = generation_result["answer"]

        retrieval_correct = score_retrieval(
            retrieved_chunks, item["source_filename"], item["expected_page"]
        )
        answer_correct, match_ratio = score_answer(generated_answer, item["expected_keywords"])

        results.append({
            "id": item["id"],
            "question": query,
            "expected_answer": item["expected_answer"],
            "generated_answer": generated_answer,
            "retrieval_correct": retrieval_correct,
            "answer_correct": answer_correct,
            "keyword_match_ratio": match_ratio,
        })

    print_summary(results)
    return results


def print_summary(results: list[dict]):
    total = len(results)
    retrieval_hits = sum(1 for r in results if r["retrieval_correct"])
    answer_hits = sum(1 for r in results if r["answer_correct"])
    avg_keyword_match = sum(r["keyword_match_ratio"] for r in results) / total if total else 0

    print("\n" + "=" * 100)
    print("EVALUATION RESULTS")
    print("=" * 100)
    print(f"{'ID':<5}{'Retrieval':<12}{'Answer':<10}{'KW Match':<10}{'Question'}")
    print("-" * 100)
    for r in results:
        retrieval_mark = "PASS" if r["retrieval_correct"] else "FAIL"
        answer_mark = "PASS" if r["answer_correct"] else "FAIL"
        print(f"{r['id']:<5}{retrieval_mark:<12}{answer_mark:<10}"
              f"{r['keyword_match_ratio']:<10.2f}{r['question'][:60]}")

    print("-" * 100)
    print(f"\nRetrieval precision: {retrieval_hits}/{total} "
          f"({retrieval_hits / total * 100:.1f}%)")
    print(f"Answer correctness:  {answer_hits}/{total} "
          f"({answer_hits / total * 100:.1f}%)")
    print(f"Avg keyword match:   {avg_keyword_match * 100:.1f}%")
    print("=" * 100)

    print("\nFailed cases (for debugging):")
    failures = [r for r in results if not r["retrieval_correct"] or not r["answer_correct"]]
    if not failures:
        print("  None — all cases passed both metrics.")
    for r in failures:
        print(f"\n  [{r['id']}] {r['question']}")
        print(f"    Expected: {r['expected_answer']}")
        print(f"    Got:      {r['generated_answer']}")
        print(f"    Retrieval correct: {r['retrieval_correct']} | "
              f"Answer correct: {r['answer_correct']}")


if __name__ == "__main__":
    run_eval()