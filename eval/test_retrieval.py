"""
Code evals: deterministic retrieval quality tests.
Run in CI: pytest eval/test_retrieval.py -v
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# Ground truth: manually annotated {query -> list[expected_chunk_ids]}
# Build this by inspecting real Pinecone results for your top queries and
# marking which chunks are genuinely relevant.
GROUND_TRUTH = {
    "how does attention work in transformers?": [
        "attention-paper-chunk-4",
        "attention-paper-chunk-7",
    ],
    "what is RLHF?": [
        "rlhf-paper-chunk-2",
        "rlhf-paper-chunk-5",
        "rlhf-paper-chunk-9",
    ],
    "how do diffusion models generate images?": [
        "diffusion-paper-chunk-1",
        "diffusion-paper-chunk-3",
    ],
}


def precision_at_k(expected_ids: list, retrieved: list) -> float:
    retrieved_ids = {c["chunk_id"] for c in retrieved}
    hits = set(expected_ids) & retrieved_ids
    return len(hits) / len(expected_ids)


@pytest.fixture(scope="session")
def retriever():
    from app.retriever import Retriever
    return Retriever()


@pytest.mark.parametrize("query,expected_ids,min_precision", [
    (
        "how does attention work in transformers?",
        GROUND_TRUTH["how does attention work in transformers?"],
        0.6,
    ),
    (
        "what is RLHF?",
        GROUND_TRUTH["what is RLHF?"],
        0.8,
    ),
    (
        "how do diffusion models generate images?",
        GROUND_TRUTH["how do diffusion models generate images?"],
        0.5,
    ),
])
def test_retrieval_precision(retriever, query, expected_ids, min_precision):
    results = retriever.search(query, k=5)
    p = precision_at_k(expected_ids, results)
    assert p >= min_precision, (
        f"Precision {p:.2f} below threshold {min_precision} for query: '{query}'"
    )


def test_metadata_completeness(retriever):
    """Every returned chunk must carry the fields the UI and citations depend on."""
    results = retriever.search("transformers", k=5)
    required = {"chunk_id", "paper_title", "section_title", "page_number", "source"}
    for chunk in results:
        missing = required - chunk.keys()
        assert not missing, f"Chunk missing required metadata fields: {missing}"


def test_returns_requested_k(retriever):
    """Search should return exactly k results when the index has enough vectors."""
    for k in (1, 3, 5):
        results = retriever.search("neural networks", k=k)
        assert len(results) == k, f"Expected {k} results, got {len(results)}"


def test_no_duplicate_chunks(retriever):
    """Duplicate chunks in a single result set corrupt context and inflate token usage."""
    results = retriever.search("large language models", k=5)
    ids = [c["chunk_id"] for c in results]
    assert len(ids) == len(set(ids)), f"Duplicate chunk_ids in results: {ids}"


def test_scores_descending(retriever):
    """Results should be ordered by relevance score, highest first."""
    results = retriever.search("BERT pre-training", k=5)
    scores = [c.get("score", 0) for c in results]
    assert scores == sorted(scores, reverse=True), "Results not sorted by score descending"
