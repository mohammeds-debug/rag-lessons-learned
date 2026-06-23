"""
Enhanced retriever with AI query improvement, context caching, and conversation memory.
"""

import hashlib
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

try:
    from app.retriever import DocumentRetriever
    from app.ai_client_factory import get_ai_client, get_client_info
    from app.sqlite_database import ConversationDB
except ImportError:
    from retriever import DocumentRetriever
    from ai_client_factory import get_ai_client, get_client_info
    from sqlite_database import ConversationDB


def _format_doc(doc: Dict[str, Any]) -> str:
    """Format a retrieved document with its metadata for use in the AI prompt."""
    meta = doc["metadata"]
    parts = []

    if meta.get("paper_title"):
        parts.append(f"Paper: {meta['paper_title']}")
    if meta.get("source"):
        parts.append(f"File: {meta['source']}")
    if meta.get("section_title") and meta["section_title"] != "Unknown":
        parts.append(f"Section: {meta['section_title']}")
    if meta.get("page_number"):
        parts.append(f"Page: {meta['page_number']}")

    header = " | ".join(parts) if parts else "Source: unknown"
    return f"{header}\n\n{doc['content']}"


class EnhancedRetriever:

    def __init__(self):
        self.document_retriever = DocumentRetriever()
        self.ai_client = get_ai_client()
        self.conversation_db = ConversationDB()

        client_info = get_client_info()
        print(f"Enhanced Retriever: {client_info['provider']} ({client_info['model']})")

    def get_session_id(self) -> str:
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

    def initialize_session(self, session_id: str) -> bool:
        return self.conversation_db.create_session(session_id, {
            "created_via": "enhanced_retriever"
        })

    @traceable(name="enhanced_retriever_query")
    def query_with_improvements(
        self,
        query: str,
        k: int = 5,
        lang: Optional[str] = None,
        use_query_improvement: bool = True,
        use_context_parsing: bool = True,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not session_id:
            session_id = self.get_session_id()

        self.initialize_session(session_id)
        self.conversation_db.add_conversation_message(
            session_id, "user", query, {"timestamp": datetime.now().isoformat()}
        )

        query_hash = hashlib.md5(query.encode()).hexdigest()

        # Check cache
        cached = self.conversation_db.get_cached_context(session_id, query_hash)
        if cached:
            self.conversation_db.add_conversation_message(
                session_id, "assistant", "Retrieved from cache", {"cached": True}
            )
            return {
                "original_query": cached["original_query"],
                "improved_query": cached["improved_query"],
                "retrieved_documents": [],
                "cached": True,
                "session_id": session_id,
                "run_id": None,
            }

        # Improve query
        improved_query = query
        if use_query_improvement:
            conv_context = self.conversation_db.get_conversation_context_for_enhancement(session_id, limit=5)
            conv_history = self.conversation_db.get_conversation_history(session_id, limit=5)
            improved_query = self.ai_client.improve_query(query, conv_history, conv_context)

        # Retrieve documents
        retrieved_docs = self.document_retriever.query_documents(
            query=improved_query, k=k, lang=lang
        )

        # Cache context
        if retrieved_docs:
            concatenated = "\n\n---\n\n".join(_format_doc(d) for d in retrieved_docs)
            self.conversation_db.cache_context(
                session_id=session_id,
                query_hash=query_hash,
                original_query=query,
                improved_query=improved_query,
                retrieved_context=concatenated,
            )

        # Generate AI response
        ai_response_text = ""
        structured_references = []
        research_area = "General ML"

        if retrieved_docs:
            conv_history = self.conversation_db.get_conversation_history(session_id, limit=5)
            concatenated = "\n\n---\n\n".join(_format_doc(d) for d in retrieved_docs)

            ai_result = self.ai_client.generate_chat_response(
                query=query,
                context=concatenated,
                conversation_history=conv_history,
            )

            ai_response_text = ai_result.get("response", "")
            structured_references = ai_result.get("references", [])
            research_area = ai_result.get("research_area", "General ML")

        self.conversation_db.add_conversation_message(
            session_id, "assistant", ai_response_text,
            {"improved_query": improved_query, "document_count": len(retrieved_docs)},
        )

        run_id = None
        try:
            current_run = get_current_run_tree()
            if current_run:
                run_id = str(current_run.id)
        except Exception:
            pass

        return {
            "original_query": query,
            "improved_query": improved_query,
            "retrieved_documents": retrieved_docs,
            "ai_response": ai_response_text,
            "structured_references": structured_references,
            "research_area": research_area,
            "cached": False,
            "session_id": session_id,
            "run_id": run_id,
        }

    def get_collection_stats(self) -> Dict[str, Any]:
        return self.document_retriever.get_collection_stats()

    def index_documents(self, data_dir: str = "/data", force_reindex: bool = False) -> int:
        return self.document_retriever.index_documents(data_dir, force_reindex)


def get_enhanced_retriever() -> EnhancedRetriever:
    return EnhancedRetriever()
