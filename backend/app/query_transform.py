"""
Query transformation techniques for improving RAG retrieval accuracy.

The vocabulary gap problem: users ask questions in natural language; papers
are written in technical language. A query like "how do models learn from
feedback?" may miss chunks that would match "reinforcement learning from
human preferences" — the same concept, different words.

Two techniques here address this at query time without rebuilding the index:

1. Multi-query retrieval — generate N phrasings, retrieve for each, merge
   with Reciprocal Rank Fusion. Broad coverage, safe default.

2. HyDE (Hypothetical Document Embeddings) — generate a short hypothetical
   paper excerpt that would answer the question, embed that instead of the
   question. The hypothetical excerpt lives in the same semantic space as
   real paper text; the raw question usually does not.

Both require an extra LLM call per user query. Use Haiku-class models to
keep latency and cost low — the output does not need to be accurate, only
plausible enough to guide retrieval.
"""
from __future__ import annotations

import anthropic

try:
    from app.hybrid_retriever import reciprocal_rank_fusion
except ImportError:
    from hybrid_retriever import reciprocal_rank_fusion

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def generate_query_variants(query: str, n: int = 3) -> list[str]:
    """
    Generate N semantically equivalent phrasings of the query.

    Returns [original_query, variant_1, ..., variant_n].
    The original query is always included — it is often the best phrasing.
    """
    response = _get_client().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                f"Generate {n} different ways to phrase this question for searching "
                f"academic AI/ML papers. Use technical vocabulary where appropriate. "
                f"Return only the questions, one per line, no numbering or labels.\n\n"
                f"Question: {query}"
            ),
        }],
    )
    lines = response.content[0].text.strip().splitlines()
    variants = [l.strip() for l in lines if l.strip()][:n]
    return [query] + variants


def multi_query_retrieve(query: str, retriever, k: int = 5, n_variants: int = 3) -> list[dict]:
    """
    Retrieve using N query phrasings, merge results with Reciprocal Rank Fusion.

    Use when: the user's phrasing is colloquial or ambiguous, or when
    initial retrieval precision is low on your eval set.

    Cost: 1 Haiku call + n_variants Pinecone queries.
    """
    variants = generate_query_variants(query, n=n_variants)
    ranked_lists = [retriever.query_documents(v, k=k * 2) for v in variants]
    merged = reciprocal_rank_fusion(ranked_lists)
    return merged[:k]


def generate_hypothetical_excerpt(query: str) -> str:
    """
    Generate a short paper excerpt that would answer the query (HyDE).

    The excerpt does not need to be factually correct — it only needs to
    be phrased like real paper text so that embedding similarity pulls
    the right chunks from the index.
    """
    response = _get_client().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                f"Write a short paragraph (3-5 sentences) from an academic AI/ML paper "
                f"that would directly answer this question. Write in formal, technical paper style. "
                f"Do not include a title or heading.\n\nQuestion: {query}"
            ),
        }],
    )
    return response.content[0].text.strip()


def hyde_retrieve(query: str, retriever, k: int = 5) -> list[dict]:
    """
    HyDE retrieval: embed a hypothetical answer instead of the raw question.

    Why this works: embedding models trained on text encode questions and
    answers into different regions of the vector space. A hypothetical
    answer is closer in embedding space to real paper excerpts than the
    original question is.

    Use when: pure dense retrieval on your eval set shows low precision
    for abstract or conceptual questions.

    Cost: 1 Haiku call + 1 Pinecone query.
    """
    hypothetical = generate_hypothetical_excerpt(query)
    return retriever.query_documents(hypothetical, k=k)


def combined_retrieve(query: str, retriever, k: int = 5) -> list[dict]:
    """
    Combine multi-query and HyDE via RRF for maximum recall.

    Recommended only when retrieval accuracy is critical and latency
    budget allows for 4 LLM calls + 4 Pinecone queries per user query.
    Measure whether the gain over hybrid search alone justifies the cost.
    """
    variants = generate_query_variants(query, n=2)
    hypothetical = generate_hypothetical_excerpt(query)

    all_queries = variants + [hypothetical]
    ranked_lists = [retriever.query_documents(q, k=k * 2) for q in all_queries]
    merged = reciprocal_rank_fusion(ranked_lists)
    return merged[:k]
