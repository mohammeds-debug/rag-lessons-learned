"""
Embedding service using Azure OpenAI.

Embeddings are always Azure OpenAI (text-embedding-3-small, 1536 dims)
regardless of which chat provider is configured. This is intentional:
the embedding model defines the vector space and must never change after
the index is built.
"""

import os
from typing import List, Optional

import httpx
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()


class EmbeddingService:

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_KEY")
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dimension = 1536

        if not self.endpoint or not self.api_key:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY must be set"
            )

        http_client = httpx.Client(
            timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=100),
        )
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version="2024-02-01",
            http_client=http_client,
        )

    def embed_texts(self, texts: List[str], batch_size: int = 128) -> List[List[float]]:
        """Embed a list of texts with batching. Zero-vectors on batch failure."""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = self.client.embeddings.create(model=self.model, input=batch)
                all_embeddings.extend(data.embedding for data in response.data)
            except Exception as e:
                print(f"Embedding batch {i // batch_size + 1} failed: {e}")
                # Graceful degradation: zero vectors return no results (correct)
                all_embeddings.extend([[0.0] * self.embedding_dimension for _ in batch])

        return all_embeddings

    def embed_single_text(self, text: str) -> List[float]:
        results = self.embed_texts([text])
        return results[0] if results else []


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
