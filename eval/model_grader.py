"""
Model grader eval: use an LLM as judge to score RAG answer quality.

Run against a golden eval set after every deployment:
    python eval/model_grader.py

Requires: ANTHROPIC_API_KEY set in environment.
"""
import json
import os
import sys
from dataclasses import dataclass

import anthropic

client = anthropic.Anthropic()

JUDGE_PROMPT = """You are evaluating the output quality of a RAG (Retrieval Augmented Generation) system.

Question asked by the user:
{question}

Context retrieved from the vector database and given to the system:
{context}

Answer produced by the system:
{answer}

Score each dimension from 1 to 5:
- faithfulness: Is every claim in the answer supported by the provided context? (5 = fully grounded, no hallucinations; 1 = answer contradicts or ignores context)
- relevance: Does the answer directly address the question? (5 = on-point; 1 = off-topic)
- completeness: Does the answer cover the key points available in the context? (5 = thorough; 1 = major gaps)

Return ONLY valid JSON with no extra text:
{{"faithfulness": <1-5>, "relevance": <1-5>, "completeness": <1-5>, "explanation": "<one concise sentence>"}}"""


@dataclass
class GradeResult:
    faithfulness: int
    relevance: int
    completeness: int
    explanation: str
    avg_score: float
    passed: bool


def grade(question: str, context: str, answer: str, passing_threshold: float = 4.0) -> GradeResult:
    """Grade a single RAG response using an LLM judge."""
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                question=question,
                context=context,
                answer=answer,
            ),
        }],
    )

    raw = json.loads(response.content[0].text)
    avg = (raw["faithfulness"] + raw["relevance"] + raw["completeness"]) / 3
    return GradeResult(
        faithfulness=raw["faithfulness"],
        relevance=raw["relevance"],
        completeness=raw["completeness"],
        explanation=raw["explanation"],
        avg_score=round(avg, 2),
        passed=avg >= passing_threshold,
    )


def run_eval_suite(
    eval_set: list,
    passing_threshold: float = 4.0,
    suite_pass_rate: float = 0.85,
) -> dict:
    """
    Run the full eval suite and return a summary report.

    eval_set format:
        [{"question": str, "context": str, "answer": str}, ...]

    Raises AssertionError if the suite pass rate falls below suite_pass_rate.
    """
    results = []
    for item in eval_set:
        result = grade(item["question"], item["context"], item["answer"], passing_threshold)
        results.append({
            "question": item["question"],
            "faithfulness": result.faithfulness,
            "relevance": result.relevance,
            "completeness": result.completeness,
            "avg_score": result.avg_score,
            "passed": result.passed,
            "explanation": result.explanation,
        })
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] avg={result.avg_score:.1f} — {item['question'][:60]}")

    passed = sum(1 for r in results if r["passed"])
    actual_rate = passed / len(results) if results else 0.0

    report = {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(actual_rate, 4),
        "threshold": passing_threshold,
        "suite_passed": actual_rate >= suite_pass_rate,
        "results": results,
    }
    return report


# ---------------------------------------------------------------------------
# Example golden eval set — replace answers with live RAG responses in CI
# ---------------------------------------------------------------------------
GOLDEN_EVAL_SET = [
    {
        "question": "How does the attention mechanism work in transformers?",
        "context": (
            "The attention mechanism computes a weighted sum of values, "
            "where the weights are determined by the compatibility of queries and keys. "
            "Scaled dot-product attention divides by sqrt(d_k) to stabilise gradients."
        ),
        "answer": (
            "The attention mechanism in transformers computes compatibility scores between "
            "queries and keys using scaled dot products (divided by sqrt(d_k)), then applies "
            "softmax to produce weights used to aggregate the values."
        ),
    },
    {
        "question": "What is RLHF and why is it used?",
        "context": (
            "Reinforcement Learning from Human Feedback (RLHF) fine-tunes language models "
            "using a reward model trained on human preference data, aligning model outputs "
            "with human values and reducing harmful or misleading responses."
        ),
        "answer": (
            "RLHF is a training technique that uses human preference rankings to train a "
            "reward model, which then guides further fine-tuning of the language model via "
            "reinforcement learning, improving alignment with human values."
        ),
    },
]


if __name__ == "__main__":
    print("Running model grader eval suite...\n")
    report = run_eval_suite(GOLDEN_EVAL_SET, passing_threshold=4.0, suite_pass_rate=0.85)

    print(f"\nResults: {report['passed']}/{report['total']} passed "
          f"({report['pass_rate']:.0%})")

    if not report["suite_passed"]:
        print("SUITE FAILED — pass rate below threshold.")
        sys.exit(1)

    print("SUITE PASSED.")
