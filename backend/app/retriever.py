"""
Pinecone retriever for arXiv RAG system.

Handles document indexing with change detection (skips unchanged files)
and vector search with optional metadata filtering.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langsmith import traceable
from pinecone import Pinecone, ServerlessSpec

try:
    from app.chunking import process_documents
    from app.embeddings import get_embedding_service
except ImportError:
    from chunking import process_documents
    from embeddings import get_embedding_service

load_dotenv()

_METADATA_DIR = ".pinecone_metadata"
_METADATA_FILE = "index_metadata.json"


class DocumentRetriever:

    def __init__(self, index_name: Optional[str] = None):
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "arxiv-rag")
        self.embedding_service = get_embedding_service()

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")

        print(f"Connecting to Pinecone index: {self.index_name}")
        self.pc = Pinecone(api_key=api_key)
        self._initialize_index()
        self.index = self.pc.Index(self.index_name)
        print(f"Connected to Pinecone index: {self.index_name}")

    def _initialize_index(self):
        existing = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing:
            print(f"Creating Pinecone index: {self.index_name} (1536 dims, cosine)")
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self.pc.describe_index(self.index_name).status["ready"]:
                print("Waiting for index to be ready...")
                time.sleep(1)
            print("Index ready.")
        else:
            print(f"Using existing index: {self.index_name}")

    # ------------------------------------------------------------------
    # Change detection
    # ------------------------------------------------------------------

    def _metadata_path(self) -> str:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, _METADATA_DIR, _METADATA_FILE)

    def _should_skip_indexing(self, data_dir: str) -> bool:
        """Return True if no PDFs have changed since last index."""
        try:
            if not os.path.exists(data_dir):
                return False
            pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
            if not pdf_files:
                return False

            current = {f: os.path.getmtime(os.path.join(data_dir, f)) for f in pdf_files}
            meta_path = self._metadata_path()

            if not os.path.exists(meta_path):
                return False

            with open(meta_path) as fh:
                stored = json.load(fh)

            if stored.get("data_dir") != data_dir:
                return False

            stored_files = stored.get("files", {})
            for fname, mtime in current.items():
                if fname not in stored_files or stored_files[fname] != mtime:
                    return False
            for fname in stored_files:
                if fname not in current:
                    return False

            return True
        except Exception as e:
            print(f"Change detection error: {e}")
            return False

    def _save_index_metadata(self, data_dir: str, document_count: int):
        try:
            pdf_files = {}
            if os.path.exists(data_dir):
                for f in os.listdir(data_dir):
                    if f.lower().endswith(".pdf"):
                        pdf_files[f] = os.path.getmtime(os.path.join(data_dir, f))

            meta = {
                "data_dir": data_dir,
                "files": pdf_files,
                "document_count": document_count,
                "indexed_at": time.time(),
            }
            meta_path = self._metadata_path()
            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
            with open(meta_path, "w") as fh:
                json.dump(meta, fh, indent=2)
        except Exception as e:
            print(f"Warning: could not save index metadata: {e}")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_documents(self, data_dir: str = "/data", force_reindex: bool = False) -> int:
        if not force_reindex and self._should_skip_indexing(data_dir):
            print("No changes detected — skipping reindex.")
            stats = self.index.describe_index_stats()
            return stats.get("total_vector_count", 0)

        print("Processing documents...")
        documents = process_documents(data_dir, chunk_size=800, chunk_overlap=150)

        if not documents:
            print("No documents to index.")
            return 0

        if force_reindex:
            print("Force reindex — deleting existing vectors...")
            try:
                self.index.delete(delete_all=True)
            except Exception as e:
                print(f"Warning: could not delete existing vectors: {e}")

        texts = [d["content"] for d in documents]
        metadatas = [d["metadata"] for d in documents]
        ids = []
        for doc in documents:
            meta = doc["metadata"]
            node_id = meta.get("node_id", f"chunk_{meta.get('chunk_id', 0)}")
            ids.append(f"{meta.get('source', 'doc')}_{node_id}")

        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_service.embed_texts(texts)

        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end = min(i + batch_size, len(documents))
            vectors = []
            for j in range(i, end):
                clean_meta = {k: v for k, v in metadatas[j].items() if v is not None}
                clean_meta["content"] = texts[j]
                vectors.append({"id": ids[j], "values": embeddings[j], "metadata": clean_meta})
            self.index.upsert(vectors=vectors)
            print(f"  Upserted batch {i // batch_size + 1}/{(len(documents) + batch_size - 1) // batch_size}")

        print(f"Indexed {len(documents)} chunks.")
        self._save_index_metadata(data_dir, len(documents))
        return len(documents)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @traceable(name="pinecone_vector_search")
    def query_documents(
        self,
        query: str,
        k: int = 5,
        lang: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_service.embed_single_text(query)
        filter_dict = {"language": {"$eq": lang}} if lang else None

        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            filter=filter_dict,
            include_metadata=True,
        )

        retrieved = []
        for i, match in enumerate(results.get("matches", [])):
            metadata = dict(match.get("metadata", {}))
            content = metadata.pop("content", "")
            retrieved.append({
                "content": content,
                "metadata": metadata,
                "score": match.get("score", 0.0),
                "distance": 1.0 - match.get("score", 0.0),
                "rank": i + 1,
            })
        return retrieved

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_documents": stats.get("total_vector_count", 0),
                "languages": [],
                "sources": [],
                "language_count": 0,
                "source_count": 0,
            }
        except Exception as e:
            return {"total_documents": 0, "error": str(e)}

    def reset_collection(self):
        self.index.delete(delete_all=True)
        print("Index reset — all vectors deleted.")


def get_retriever() -> DocumentRetriever:
    return DocumentRetriever()
