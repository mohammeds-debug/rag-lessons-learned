"""
Cross-encoder re-ranking for RAG retrieval.

A bi-encoder (used by Pinecone) embeds query and document independently,
then compares vectors — fast, but misses fine-grained token interactions.

A cross-encoder sees both query and document together, producing a much more
accurate relevance score — but O(k) inference calls, so only run it on a
small candidate set (top-20 from the retriever, return top-5).

Model choice:
  cross-encoder/ms-marco-MiniLM-L-6-v2  — ~80MB, fast, good for English QA
  cross-encoder/ms-marco-MiniLM-L-12-v2 — ~120MB, slightly better quality
  BAAI/bge-reranker-base                — stronger on multilingual / domain text

Install: pip install sentence-transformers
"""
from __future__ import annotations

from typing import Any

_model = None
_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_reranker():
    global _model
    if _model is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers") from exc
        _model = CrossEncoder(_model_name)
    return _model


def rerank(query: str, chunks: list[dict[str, Any]], top_n: int = 5) -> list[dict[str, Any]]:
    """
    Re-rank retrieval candidates using a cross-encoder.

    Typical call pattern:
        candidates = retriever.query_documents(query, k=20)   # fetch broadly
        final = rerank(query, candidates, top_n=5)            # narrow precisely

    The cross-encoder is not a replacement for the initial retrieval — it is
    a precision layer on top. Without the initial retrieval to narrow the
    candidate pool, scoring every chunk in the index would be too slow.
    """
    if not chunks:
        return chunks

    reranker = get_reranker()
    pairs = [(query, chunk.get("content", "")) for chunk in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked[:top_n]]


def rerank_with_scores(query: str, chunks: list[dict[str, Any]], top_n: int = 5) -> list[dict[str, Any]]:
    """
    Same as rerank() but attaches the cross-encoder score to each returned chunk
    under the key "rerank_score". Useful for debugging and eval comparison.
    """
    if not chunks:
        return chunks

    reranker = get_reranker()
    pairs = [(query, chunk.get("content", "")) for chunk in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    result = []
    for score, chunk in ranked[:top_n]:
        result.append({**chunk, "rerank_score": float(score)})
    return result
