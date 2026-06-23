"""
FastAPI backend for arXiv RAG — semantic search over AI/ML research papers.
"""

import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.enhanced_retriever import EnhancedRetriever
    from app.retriever import DocumentRetriever
    from app.sqlite_database import ConversationDB
    from app.retry_util import retry_with_exponential_backoff
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
    from enhanced_retriever import EnhancedRetriever
    from retriever import DocumentRetriever
    from sqlite_database import ConversationDB
    from retry_util import retry_with_exponential_backoff

load_dotenv()

DEFAULT_SEARCH_RESULTS_K = int(os.getenv("DEFAULT_SEARCH_RESULTS_K", "5"))
MAX_SEARCH_RESULTS_K = int(os.getenv("MAX_SEARCH_RESULTS_K", "20"))

enhanced_retriever = None
basic_retriever = None
conversation_db = None


# ── Pydantic models ────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: Optional[int] = Field(default=None, ge=1, le=100)
    lang: Optional[str] = None
    use_enhanced_features: bool = True
    use_query_improvement: bool = True
    use_context_parsing: bool = True

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

    @field_validator("k")
    @classmethod
    def validate_k_max(cls, v):
        if v is not None and v > MAX_SEARCH_RESULTS_K:
            raise ValueError(f"k must not exceed {MAX_SEARCH_RESULTS_K}")
        return v


class IndexRequest(BaseModel):
    data_dir: str = Field(default="/data")
    force_reindex: bool = False


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, bool]


class SessionCreateResponse(BaseModel):
    session_id: str
    message: str


class SessionDeleteResponse(BaseModel):
    message: str


class ConversationHistoryResponse(BaseModel):
    history: List[Dict[str, Any]]
    session_id: str


class SessionsListResponse(BaseModel):
    sessions: List[Dict[str, Any]]


class StatusResponse(BaseModel):
    indexed: bool
    message: Optional[str] = None
    collection_count: Optional[int] = None
    collection_stats: Optional[Dict[str, Any]] = None
    session_stats: Optional[Dict[str, Any]] = None
    enhanced_features_available: Optional[bool] = None


class IndexResponse(BaseModel):
    message: str
    indexed_count: int
    total_documents: int
    languages: List[str]
    sources: List[str]
    data_dir: str
    force_reindex: bool


class SettingsResponse(BaseModel):
    enhanced_features_available: bool
    default_k: int
    max_k: int
    supported_languages: List[str]
    features: Dict[str, bool]
    auto_index_enabled: bool
    data_directory: str


class FeedbackRequest(BaseModel):
    run_id: Optional[str] = None
    session_id: str
    message_index: int
    rating: str
    comment: Optional[str] = None
    query: Optional[str] = None

    @field_validator("rating")
    @classmethod
    def rating_must_be_valid(cls, v):
        if v not in ["positive", "negative"]:
            raise ValueError('Rating must be "positive" or "negative"')
        return v


class FeedbackResponse(BaseModel):
    message: str
    feedback_id: Optional[str] = None


# ── Startup / shutdown ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting arXiv RAG backend...")
    if not await initialize_services():
        print("Service initialization failed.")
        sys.exit(1)
    print("Services ready.")
    yield
    print("Shutting down arXiv RAG backend...")


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="arXiv RAG API",
    description="Semantic search over AI/ML research papers using RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins == "*":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in allowed_origins.split(",")],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


async def initialize_services() -> bool:
    global enhanced_retriever, basic_retriever, conversation_db
    try:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(parent_dir)

        basic_retriever = DocumentRetriever()
        enhanced_retriever = EnhancedRetriever()
        conversation_db = ConversationDB()

        auto_index = os.getenv("AUTO_INDEX_ON_STARTUP", "false").lower() == "true"
        data_dir = os.getenv("DATA_DIR", "/data")

        if auto_index and os.path.exists(data_dir):
            pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
            if pdf_files:
                print(f"Auto-indexing {len(pdf_files)} PDF files...")
                count = basic_retriever.index_documents(data_dir, force_reindex=False)
                print(f"Auto-indexed {count} chunks.")

        return True
    except Exception as e:
        print(f"Initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_session_id(x_session_id: Optional[str] = None) -> str:
    if x_session_id:
        return x_session_id
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"


@retry_with_exponential_backoff(max_retries=3, initial_delay=1.0, max_delay=8.0)
def perform_search_with_retry(search_request: SearchRequest, session_id: str) -> Dict[str, Any]:
    if not basic_retriever:
        raise RuntimeError("Retriever not initialized")

    k_value = search_request.k if search_request.k is not None else DEFAULT_SEARCH_RESULTS_K

    if search_request.use_enhanced_features and enhanced_retriever:
        conversation_db.create_session(session_id, {"created_via": "api"})
        results = enhanced_retriever.query_with_improvements(
            query=search_request.query,
            k=k_value,
            lang=search_request.lang,
            use_query_improvement=search_request.use_query_improvement,
            use_context_parsing=search_request.use_context_parsing,
            session_id=session_id,
        )
        results["session_id"] = session_id
    else:
        retrieved_docs = basic_retriever.query_documents(
            query=search_request.query, k=k_value, lang=search_request.lang
        )
        results = {
            "original_query": search_request.query,
            "improved_query": search_request.query,
            "retrieved_documents": retrieved_docs,
            "ai_response": None,
            "cached": False,
            "session_id": session_id,
            "run_id": None,
        }

    return results


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        services={
            "basic_retriever": basic_retriever is not None,
            "enhanced_retriever": enhanced_retriever is not None,
            "conversation_db": conversation_db is not None,
        },
    )


@app.get("/api/status", response_model=StatusResponse, tags=["System"])
async def get_status(x_session_id: Optional[str] = Header(None)):
    if not basic_retriever:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    try:
        stats = basic_retriever.get_collection_stats()
        count = stats.get("total_documents", 0)
        indexed = count > 0
    except Exception:
        indexed, count = False, 0

    if not indexed:
        return StatusResponse(
            indexed=False,
            message="No documents indexed. Run the indexing script first.",
            collection_count=count,
        )

    session_stats = None
    if enhanced_retriever and x_session_id:
        try:
            session_stats = conversation_db.get_session_stats(x_session_id)
        except Exception:
            pass

    return StatusResponse(
        indexed=True,
        collection_stats=stats,
        session_stats=session_stats,
        enhanced_features_available=enhanced_retriever is not None,
    )


@app.post("/api/search", tags=["Search"])
@limiter.limit("8/minute")
async def search_documents(
    request: Request,
    search_request: SearchRequest,
    x_session_id: Optional[str] = Header(None),
):
    """Search papers with AI enhancement. Rate limited to 8 req/min per IP."""
    try:
        session_id = get_session_id(x_session_id)
        results = perform_search_with_retry(search_request, session_id)
        return results
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Search temporarily unavailable.")


@app.get("/api/conversation/history", response_model=ConversationHistoryResponse, tags=["Conversation"])
async def get_conversation_history(
    x_session_id: Optional[str] = Header(None),
    limit: int = Query(default=10, ge=1, le=100),
):
    session_id = get_session_id(x_session_id)
    if not conversation_db:
        raise HTTPException(status_code=500, detail="Conversation DB not initialized")
    history = conversation_db.get_conversation_history(session_id, limit)
    return ConversationHistoryResponse(history=history, session_id=session_id)


@app.get("/api/conversation/sessions", response_model=SessionsListResponse, tags=["Conversation"])
async def get_conversation_sessions():
    if not conversation_db:
        raise HTTPException(status_code=500, detail="Conversation DB not initialized")
    return SessionsListResponse(sessions=conversation_db.get_all_sessions())


@app.get("/api/conversation/session/{session_id}", response_model=ConversationHistoryResponse, tags=["Conversation"])
async def get_session_history(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=100),
):
    if not conversation_db:
        raise HTTPException(status_code=500, detail="Conversation DB not initialized")
    history = conversation_db.get_conversation_history(session_id, limit)
    return ConversationHistoryResponse(history=history, session_id=session_id)


@app.delete("/api/conversation/session/{session_id}", response_model=SessionDeleteResponse, tags=["Conversation"])
async def delete_session(session_id: str):
    if not conversation_db:
        raise HTTPException(status_code=500, detail="Conversation DB not initialized")
    success = conversation_db.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete session")
    return SessionDeleteResponse(message="Session deleted")


@app.post("/api/conversation/new", response_model=SessionCreateResponse, tags=["Conversation"])
async def create_new_session():
    if not conversation_db:
        raise HTTPException(status_code=500, detail="Conversation DB not initialized")
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    success = conversation_db.create_session(session_id, {"created_at": datetime.now().isoformat()})
    if not success:
        raise HTTPException(status_code=500, detail="Failed to create session")
    return SessionCreateResponse(session_id=session_id, message="New session created")


@app.post("/api/index", response_model=IndexResponse, tags=["Indexing"])
@limiter.limit("3/minute")
async def index_documents(request: Request, index_request: IndexRequest = None):
    """Trigger document indexing. Rate limited to 3 req/min per IP."""
    if not basic_retriever:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    if index_request is None:
        index_request = IndexRequest()

    if not os.path.exists(index_request.data_dir):
        raise HTTPException(status_code=400, detail=f"Directory {index_request.data_dir} not found")

    pdf_files = [f for f in os.listdir(index_request.data_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise HTTPException(status_code=400, detail="No PDF files found")

    count = basic_retriever.index_documents(index_request.data_dir, index_request.force_reindex)
    stats = basic_retriever.get_collection_stats()

    return IndexResponse(
        message=f"Indexed {count} chunks",
        indexed_count=count,
        total_documents=stats["total_documents"],
        languages=stats.get("languages", []),
        sources=stats.get("sources", []),
        data_dir=index_request.data_dir,
        force_reindex=index_request.force_reindex,
    )


@app.get("/api/settings", response_model=SettingsResponse, tags=["System"])
async def get_settings():
    return SettingsResponse(
        enhanced_features_available=enhanced_retriever is not None,
        default_k=DEFAULT_SEARCH_RESULTS_K,
        max_k=MAX_SEARCH_RESULTS_K,
        supported_languages=["en"],
        features={
            "query_improvement": enhanced_retriever is not None,
            "context_parsing": enhanced_retriever is not None,
            "conversation_memory": conversation_db is not None,
            "context_caching": conversation_db is not None,
        },
        auto_index_enabled=os.getenv("AUTO_INDEX_ON_STARTUP", "false").lower() == "true",
        data_directory=os.getenv("DATA_DIR", "/data"),
    )


@app.post("/api/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(feedback_request: FeedbackRequest):
    """Submit feedback linked to a LangSmith run."""
    try:
        from langsmith import Client
        langsmith_client = Client()
        score = 1.0 if feedback_request.rating == "positive" else 0.0
        comment_parts = []
        if feedback_request.comment:
            comment_parts.append(feedback_request.comment)
        comment_parts.append(f"Session: {feedback_request.session_id}")
        comment_parts.append(f"Message: {feedback_request.message_index}")
        if feedback_request.query:
            comment_parts.append(f"Query: {feedback_request.query}")

        if feedback_request.run_id:
            fb = langsmith_client.create_feedback(
                run_id=feedback_request.run_id,
                key="user_rating",
                score=score,
                comment=" | ".join(comment_parts),
                value=feedback_request.rating,
            )
            return FeedbackResponse(message="Feedback submitted", feedback_id=str(fb.id))
        else:
            return FeedbackResponse(message="Feedback received (no run_id to link)", feedback_id=None)
    except Exception as e:
        return FeedbackResponse(message=f"Feedback recorded locally: {e}", feedback_id=None)


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(status_code=404, content={"error": "Endpoint not found"})


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=os.getenv("FASTAPI_ENV") != "production")
