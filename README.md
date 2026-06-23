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

---

## Production Lessons

This codebase was designed around 12 specific lessons learned from shipping a production RAG system. Each decision is traceable to a real problem:

### 1. Lock your embedding model on day one

`text-embedding-3-small` (1536 dims) is hardcoded in `embeddings.py` and never changes, regardless of which chat provider you configure. Switching embedding models requires rebuilding the entire index. The chat layer (Gemini ↔ Azure OpenAI) can be swapped freely via `CLIENT_TO_BE_USED` because it doesn't affect the vector space.

**File**: `backend/app/embeddings.py`, `backend/app/ai_client_factory.py`

---

### 2. Change detection prevents redundant re-indexing

Before processing any PDF, the indexer compares modification timestamps against a stored manifest (`.pinecone_metadata/index_metadata.json`). Unchanged files are skipped. Only new or modified PDFs are re-embedded and upserted.

Without this: every restart with `AUTO_INDEX_ON_STARTUP=true` re-indexes the full corpus, burning embedding API credits.

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

## License

MIT — use freely, attribution appreciated.

---

*Built to demonstrate production RAG engineering patterns. See the [blog post](docs/production-rag-lessons.md) for the full write-up of lessons learned.*
