The main purpose of this Repo is to get an understanding of lessons learnt during the development of a production-ready RAG application.

---

## Production Lessons

This codebase was designed around 12 specific lessons learned from shipping a production RAG system. Each decision is traceable to a real problem:

### 1. Lock your embedding model on day one

`text-embedding-3-small` (1536 dims) is hardcoded in `embeddings.py` and never changes, regardless of which chat provider you configure. Switching embedding models requires rebuilding the entire index. The chat layer (Gemini ↔ Azure OpenAI) can be swapped freely via `CLIENT_TO_BE_USED` because it doesn't affect the vector space.

**File**: `backend/app/embeddings.py`, `backend/app/ai_client_factory.py`

---

### 2. Change detection prevents redundant re-indexing

Before processing any PDF, the indexer compares modification timestamps against a stored manifest (`.pinecone_metadata/index_metadata.json`). Unchanged files are skipped. Only new or modified PDFs are re-embedded and upserted.

Without this, every restart with `AUTO_INDEX_ON_STARTUP=true` re-indexes the full corpus, burning embedding API credits.

**File**: `backend/app/retriever.py` → `_should_skip_indexing()`

---

### 3. Query improvement and response generation need different LLM budgets

Two different pipeline stages, two different configurations:

- **Query improvement** (`thinking_budget=1024`): "ViT" → "Vision Transformer image classification self-attention patch embeddings" — needs reasoning to expand acronyms and map user intent to paper terminology
- **Response generation** (`thinking_budget=0`): context is already retrieved; just format and cite it — reasoning adds latency and cost with no quality gain

**File**: `backend/app/gemini_client.py` → `improve_query()`, `generate_chat_response()`

---

### 4. Cache retrieved context, not generated responses

The expensive steps are vector search + query improvement (LLM API call). The response generation is fast and needs fresh conversation context anyway.

Cache key: `MD5(query) + session_id` → stored in SQLite. On cache hit, Pinecone and the query improver are bypassed entirely. Response generation still runs (incorporating fresh conversation history).

**File**: `backend/app/enhanced_retriever.py`, `backend/app/sqlite_database.py`

---

### 5. Enrich metadata at index time, exhaustively

Every chunk stored in Pinecone carries: `paper_title`, `section_title`, `page_number`, `source`, `chunk_id`, `language`. This metadata is extracted once at index time (`chunking.py`) and returned with every search result — no secondary lookups needed.

The UI displays "Attention Is All You Need | Section: Results | p. 8" directly from Pinecone metadata.

**File**: `backend/app/chunking.py` → `_build_chunks()`

---

### 6. Prompt injection is a RAG-specific attack surface

A RAG pipeline has multiple LLM calls: query improver → response generator. A malicious query can propagate through the chain. Two defenses:

1. **Pattern matching** (30+ patterns): "ignore previous instructions", "DAN", "jailbreak", etc. — checked before any LLM call
2. **Boundary markers**: every piece of user-controlled content is wrapped in `[USER_QUERY_START]...[USER_QUERY_END]` so the model treats it as data, not instructions

**File**: `backend/app/prompt_utils.py`, called from both AI clients

---

### 7. Async is not optional for RAG backends

Every search request makes 3+ external API calls (Pinecone + query improver + response generator). In a synchronous framework (Flask), each concurrent request blocks a thread for the full duration of all I/O. Under load, the thread pool exhausts quickly.

FastAPI with `async/await` throughout means the server handles concurrent requests without proportionally growing thread count.

**File**: `backend/server.py`

---

### 8. Exponential backoff on every external API call

AI APIs have transient failures. The search function is wrapped with a decorator that retries up to 3 times with delays of 1s → 2s → 4s. The client-side timeout (60s) accommodates this.

Generic decorator works on both sync and async functions.

**File**: `backend/app/retry_util.py`, applied in `backend/server.py`

---

### 9. Chunk size is domain-specific — test it on your corpus

Final configuration: `chunk_size=800, chunk_overlap=150` (18.75% overlap).

For dense academic papers, smaller chunks lose the explanatory context that gives them meaning. Larger chunks risk exceeding the context window when retrieving k=5. The 18.75% overlap ensures sentences at chunk boundaries appear in adjacent chunks.

**Recommendation**: test retrieval quality on 20-30 real queries over your domain before committing to a chunk size.

**File**: `backend/app/retriever.py` → `index_documents()`

---

### 10. Cap conversation history at 3-5 turns

Full conversation history bloats the prompt and pushes retrieved context down — or out. Three turns is sufficient for follow-up questions ("tell me more", "what about the results section?").

```python
for msg in conversation_history[-3:]:  # Hard cap
```

**File**: `backend/app/gemini_client.py` → `improve_query()`

---

### 11. Use RAG-native observability tooling

LangSmith was chosen over MLflow specifically because it understands retrieval pipelines. Each stage is a named trace:

```python
@traceable(name="pinecone_vector_search")
@traceable(name="gemini_improve_query")
@traceable(name="enhanced_retriever_query")
```

User feedback (thumbs up/down) is linked to specific run IDs via `langsmith_client.create_feedback()`, so you can trace a bad response back to exactly which retrieval step failed.

**File**: `backend/app/retriever.py`, `backend/app/gemini_client.py`, `backend/server.py`

---

### 12. Separate "container running" from "container ready"

The `/api/health` endpoint checks that all service instances (retriever, enhanced retriever, conversation DB) finished initializing — not just that the process started. The Docker healthcheck polls this endpoint with a `start_period: 40s` grace period.

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:5000/api/health"]
  start_period: 40s  # Gives services time to initialize
```

**File**: `docker-compose.yml`, `backend/server.py` → `health_check()`

---

### 13. Evals: Code, model graders, and human-in-the-loop are non-negotiable

AI applications don't have deterministic outputs — the same query can return subtly different answers depending on context window changes, model updates, or upstream data drift. Without evals, you have no signal whether things got better or worse after a change.

Three layers are required:

**Code evals** — deterministic, fast, run in CI. Verify retrieval precision, chunk integrity, cache correctness, and injection detection with standard unit tests. These tell you your pipeline is wired correctly.

**Model graders** — use a stronger LLM as a judge to score faithfulness (no hallucinations), relevance, and completeness across a golden eval set. Run them after every deployment. They catch quality regressions that unit tests can't see.

**Human-in-the-loop (HITL)** — wire real user feedback (thumbs up/down, free-text corrections) directly into LangSmith or your eval store. This is the only ground truth you actually trust long-term. Every thumbs-down is a labelled failure case; collect enough and you have a regression suite built from production traffic.

Running only one or two of these layers will leave a blind spot. A passing unit suite won't catch a model that answers correctly but cites the wrong paper. A perfect model-grader score won't catch a systematic bug in your chunker that only users notice.

#### Code eval — retrieval precision

```python
# eval/test_retrieval.py
import pytest
from backend.app.retriever import Retriever

# Ground truth: manually annotated {query -> list[expected_chunk_ids]}
GROUND_TRUTH = {
    "how does attention work in transformers?": ["attention-paper-chunk-4", "attention-paper-chunk-7"],
    "what is RLHF?": ["rlhf-paper-chunk-2", "rlhf-paper-chunk-5", "rlhf-paper-chunk-9"],
}

def precision_at_k(expected_ids: list[str], retrieved: list[dict]) -> float:
    retrieved_ids = {c["chunk_id"] for c in retrieved}
    hits = set(expected_ids) & retrieved_ids
    return len(hits) / len(expected_ids)

@pytest.fixture(scope="session")
def retriever():
    return Retriever()

@pytest.mark.parametrize("query,expected_ids,min_precision", [
    ("how does attention work in transformers?", GROUND_TRUTH["how does attention work in transformers?"], 0.6),
    ("what is RLHF?", GROUND_TRUTH["what is RLHF?"], 0.8),
])
def test_retrieval_precision(retriever, query, expected_ids, min_precision):
    results = retriever.search(query, k=5)
    p = precision_at_k(expected_ids, results)
    assert p >= min_precision, f"Precision {p:.2f} below threshold {min_precision} for: {query}"

def test_metadata_completeness(retriever):
    """Every chunk must carry the fields the UI depends on."""
    results = retriever.search("transformers", k=3)
    required = {"chunk_id", "paper_title", "section_title", "page_number", "source"}
    for chunk in results:
        missing = required - chunk.keys()
        assert not missing, f"Chunk missing fields: {missing}"
```

#### Model grader eval — LLM as judge

```python
# eval/model_grader.py
import json
import anthropic

client = anthropic.Anthropic()

JUDGE_PROMPT = """You are evaluating the output of a RAG system.

Question: {question}
Retrieved context: {context}
System answer: {answer}

Score each dimension 1–5:
- faithfulness: Is every claim in the answer supported by the context? (5 = fully grounded, no hallucinations)
- relevance: Does the answer address what was asked? (5 = directly answers the question)
- completeness: Does it cover the key points available in the context? (5 = thorough)

Return only valid JSON: {{"faithfulness": N, "relevance": N, "completeness": N, "explanation": "one sentence"}}"""

def grade(question: str, context: str, answer: str) -> dict:
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(question=question, context=context, answer=answer)
        }]
    )
    return json.loads(response.content[0].text)

def run_eval_suite(eval_set: list[dict], passing_threshold: float = 4.0) -> dict:
    """
    eval_set: list of {"question": str, "context": str, "answer": str}
    Returns summary with per-question scores and overall pass/fail.
    """
    results = []
    for item in eval_set:
        scores = grade(item["question"], item["context"], item["answer"])
        avg = sum([scores["faithfulness"], scores["relevance"], scores["completeness"]]) / 3
        results.append({**item, **scores, "avg_score": avg, "passed": avg >= passing_threshold})

    passed = sum(1 for r in results if r["passed"])
    return {
        "total": len(results),
        "passed": passed,
        "pass_rate": passed / len(results),
        "results": results,
    }

# Usage:
# suite = [
#     {"question": "How does attention work?", "context": "<retrieved chunks>", "answer": "<rag answer>"},
# ]
# report = run_eval_suite(suite)
# assert report["pass_rate"] >= 0.85, f"Eval suite failed: {report['pass_rate']:.0%} pass rate"
```

#### Human-in-the-loop eval — wiring feedback to LangSmith

```python
# backend/app/hitl_eval.py
from langsmith import Client

langsmith_client = Client()

def record_user_feedback(run_id: str, thumbs_up: bool, comment: str = "") -> None:
    """Called from the /api/feedback endpoint — links UI feedback to the exact RAG trace."""
    langsmith_client.create_feedback(
        run_id=run_id,
        key="user_satisfaction",
        score=1 if thumbs_up else 0,
        comment=comment,
        feedback_source_type="app",
    )

def export_negative_feedback_as_eval_set(limit: int = 100) -> list[dict]:
    """
    Pull thumbs-down runs from LangSmith and turn them into a labelled eval set.
    Run this periodically to grow your golden regression suite from real failures.
    """
    feedback_list = langsmith_client.list_feedback(
        feedback_key=["user_satisfaction"],
        limit=limit,
    )

    eval_cases = []
    for fb in feedback_list:
        if fb.score == 0:  # thumbs down only
            run = langsmith_client.read_run(fb.run_id)
            eval_cases.append({
                "run_id": str(fb.run_id),
                "question": run.inputs.get("query", ""),
                "answer": run.outputs.get("response", "") if run.outputs else "",
                "user_comment": fb.comment or "",
            })

    return eval_cases

# Suggested CI workflow:
# 1. On each PR, run test_retrieval.py (code evals) — must pass to merge.
# 2. After deploy to staging, run run_eval_suite() against your golden set — alert if pass_rate drops.
# 3. Weekly: export_negative_feedback_as_eval_set() → review → promote good cases to the golden set.
```

**File**: `eval/test_retrieval.py`, `eval/model_grader.py`, `backend/app/hitl_eval.py`

---

# Sample Project 

# arXiv RAG

Production-ready Retrieval Augmented Generation (RAG) system for semantic search over AI/ML research papers.

Ask questions in natural language and get answers grounded in specific papers — with citations, section references, and page numbers.

---

## Demo Questions

- *"How does the attention mechanism work in transformers?"*
- *"What is RLHF and how does it align language models?"*
- *"How do diffusion models generate images?"*
- *"What are the key differences between BERT and GPT?"*
- *"Explain RAG systems and how they reduce hallucination"*

---

## Architecture

```
User Query
    │
    ▼
Query Improvement (Gemini/GPT)     ← expands "ViT" → "Vision Transformer..."
    │
    ▼
Vector Search (Pinecone)           ← cosine similarity over 1536-dim embeddings
    │
    ▼
Retrieved Paper Excerpts           ← with paper_title, section, page_number
    │
    ▼
Response Generation (Gemini/GPT)   ← grounded answer with citations
    │
    ▼
React Frontend                     ← typing effect, feedback buttons, session history
```

**Stack:**
- **Frontend**: React, Axios, react-markdown
- **Backend**: FastAPI (async), Python 3.11
- **Embeddings**: Azure OpenAI `text-embedding-3-small` (1536 dims, fixed)
- **Vector DB**: Pinecone (serverless, AWS us-east-1)
- **AI Providers**: Google Gemini (default) or Azure OpenAI (configurable)
- **Session DB**: SQLite (conversation history + query cache)
- **Observability**: LangSmith tracing (optional)

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/your-username/arxiv-rag.git
cd arxiv-rag
cp .env.example .env
# Fill in your API keys in .env
```

### 2. Download papers

```bash
pip install arxiv
python scripts/fetch_papers.py --count 30
# Downloads 30 papers from cs.AI, cs.LG, cs.CL into ./data/
```

Or use a custom query:
```bash
python scripts/fetch_papers.py --query "large language models instruction tuning" --count 20
python scripts/fetch_papers.py --categories cs.CV cs.RO --count 25
```

### 3. Index documents

```bash
cd backend
pip install -r requirements.txt
python index_documents.py
```

### 4. Start the application

**Docker (recommended):**
```bash
docker-compose up -d
# Frontend: http://localhost:80
# Backend:  http://localhost:5001
```

**Local development:**
```bash
# Terminal 1 — Backend
cd backend
uvicorn server:app --host 0.0.0.0 --port 5000 --reload

# Terminal 2 — Frontend
cd frontend
npm install
npm start  # http://localhost:3000
```

### 5. Verify

```bash
curl http://localhost:5001/api/health
curl http://localhost:5001/api/status
```

---

## Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Yes | Azure OpenAI endpoint for embeddings |
| `AZURE_OPENAI_KEY` | Yes | Azure OpenAI API key |
| `PINECONE_API_KEY` | Yes | Pinecone vector DB key |
| `PINECONE_INDEX_NAME` | Yes | Index name (default: `arxiv-rag`) |
| `CLIENT_TO_BE_USED` | Yes | `gemini` or `openai` |
| `GEMINI_API_KEY` | If Gemini | Google Gemini API key |
| `AZURE_AI_CHAT_*` | If OpenAI | Azure OpenAI chat credentials |
| `AUTO_INDEX_ON_STARTUP` | No | Auto-index PDFs on startup (default: false) |
| `LANGCHAIN_API_KEY` | No | LangSmith tracing (optional) |

**Switching AI providers** requires only a config change — no code changes:
```bash
CLIENT_TO_BE_USED=gemini   # Google Gemini (default, cost-effective)
CLIENT_TO_BE_USED=openai   # Azure OpenAI
```

**Note**: The embedding model (`AZURE_OPENAI_ENDPOINT` + `text-embedding-3-small`) is fixed and cannot be changed after indexing without rebuilding the entire index.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/status` | Index status and document count |
| POST | `/api/search` | Semantic search with AI enhancement |
| POST | `/api/conversation/new` | Create new session |
| GET | `/api/conversation/sessions` | List all sessions |
| GET | `/api/conversation/session/{id}` | Get session history |
| DELETE | `/api/conversation/session/{id}` | Delete session |
| POST | `/api/index` | Trigger manual indexing |
| GET | `/api/settings` | System configuration |
| POST | `/api/feedback` | Submit feedback (linked to LangSmith) |

Search request body:
```json
{
  "query": "how does attention work in transformers?",
  "k": 5,
  "use_enhanced_features": true,
  "use_query_improvement": true
}
```

---

## Adding Your Own Papers

1. Place PDF files in `./data/`
2. Run indexing:
   ```bash
   python backend/index_documents.py
   # or force a full reindex:
   python backend/index_documents.py --force
   ```
3. No restart required — new vectors are available immediately

## Project Structure

```
arxiv-rag/
├── backend/
│   ├── app/
│   │   ├── ai_client_factory.py  # Provider switching (Gemini ↔ Azure OpenAI)
│   │   ├── chunking.py           # PDF extraction + section-aware chunking
│   │   ├── embeddings.py         # Azure OpenAI embeddings (fixed provider)
│   │   ├── enhanced_retriever.py # Query improvement + caching + session memory
│   │   ├── gemini_client.py      # Google Gemini integration
│   │   ├── llm_client.py         # Azure OpenAI chat integration
│   │   ├── prompt_utils.py       # Injection detection + boundary markers
│   │   ├── retriever.py          # Pinecone indexing + vector search
│   │   ├── retry_util.py         # Exponential backoff decorator
│   │   └── sqlite_database.py    # Sessions, messages, context cache
│   ├── server.py                 # FastAPI app, rate limiting, endpoints
│   └── index_documents.py        # CLI indexing script
├── frontend/
│   └── src/
│       ├── App.js                # Main app with session management
│       ├── components/           # SearchInput, SearchResults, etc.
│       ├── hooks/useTypingEffect.js
│       └── services/api.js       # Axios API client
├── scripts/
│   └── fetch_papers.py           # arXiv paper downloader
├── data/                         # PDF files (gitignored)
├── docker-compose.yml
├── .env.example
└── README.md
```

---
