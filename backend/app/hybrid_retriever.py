"""
Hybrid retrieval: combines Pinecone dense search with BM25 sparse search.

Merge strategy: Reciprocal Rank Fusion (RRF) — no score normalisation needed,
works across arbitrary scoring scales.

Usage:
    from app.hybrid_retriever import HybridRetriever, build_bm25_corpus

    corpus = build_bm25_corpus(retriever)          # call once at startup
    hybrid = HybridRetriever(retriever, corpus)
    results = hybrid.search("attention mechanism", k=5)
"""
from __future__ import annotations

from typing import Any

try:
    from rank_bm25 import BM25Okapi
except ImportError as exc:
    raise ImportError("Install rank-bm25: pip install rank-bm25") from exc

try:
    from app.retriever import DocumentRetriever
except ImportError:
    from retriever import DocumentRetriever


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Merge N ranked result lists into one using Reciprocal Rank Fusion.

    RRF score for a document d across lists: sum(1 / (k + rank(d, list)))
    k=60 is the standard default — it dampens the advantage of top-rank
    positions without ignoring them.

    No score normalisation is required, making this safe to use across
    dense similarity scores and BM25 scores which are on different scales.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            cid = chunk.get("metadata", {}).get("chunk_id", chunk.get("chunk_id", str(id(chunk))))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = chunk

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    return [chunk_map[cid] for cid in sorted_ids]


class HybridRetriever:
    """
    Wraps a DocumentRetriever (dense) and a BM25 index (sparse).
    Merges results from both using Reciprocal Rank Fusion.

    When to prefer hybrid over pure dense:
    - Queries containing exact acronyms ("ViT", "BERT", "PPO")
    - Queries using author names or paper titles
    - Queries with rare technical terms not well-represented in the embedding space
    """

    def __init__(self, dense_retriever: DocumentRetriever, corpus_chunks: list[dict[str, Any]]):
        self.dense = dense_retriever
        self.chunks = corpus_chunks

        tokenized = [self._tokenize(c.get("content", "")) for c in corpus_chunks]
        self.bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def search(self, query: str, k: int = 5, fetch_k: int = 20) -> list[dict[str, Any]]:
        """
        Retrieve top-k chunks using hybrid dense + sparse search.

        fetch_k: how many candidates each method retrieves before merging.
                 Should be 3-4x k to give RRF enough material to re-rank.
        """
        # Dense retrieval via Pinecone
        dense_results = self.dense.query_documents(query, k=fetch_k)

        # Sparse retrieval via BM25
        bm25_scores = self.bm25.get_scores(self._tokenize(query))
        top_sparse_idx = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:fetch_k]
        sparse_results = [self.chunks[i] for i in top_sparse_idx]

        merged = reciprocal_rank_fusion([dense_results, sparse_results])
        return merged[:k]


def build_bm25_corpus(retriever: DocumentRetriever, data_dir: str = "/data") -> list[dict[str, Any]]:
    """
    Pull all indexed chunks from Pinecone to build the BM25 in-memory index.
    Call once at startup; re-call after re-indexing.

    For very large corpora (>100k chunks), consider persisting the BM25 index to disk.
    """
    from app.chunking import process_documents
    documents = process_documents(data_dir, chunk_size=800, chunk_overlap=150)
    return documents
