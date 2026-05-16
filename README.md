# SHL Assessment Recommender

A conversational AI agent that recommends SHL assessments through multi-turn dialogue. Built with FastAPI, FAISS semantic search, and LLM-powered orchestration.

**Live API:** [Deployed on Railway / Render](https://shl-assessment-recommender.onrender.com)  
**GitHub:** [Yashmaggu-NV/shl-assessment-recommender](https://github.com/Yashmaggu-NV/shl-assessment-recommender)

---

## Features

- **Multi-turn conversation** — Clarifies vague queries, refines shortlists, compares assessments
- **SHL catalog-grounded** — Only recommends real SHL products with verified URLs
- **Zero hallucination** — All names, URLs, and metadata come strictly from the catalog
- **Balanced batteries** — Composes assessment sets across Knowledge, Cognitive, Personality, and SJT categories
- **Strict domain filtering** — Prevents drift into unrelated domains (sales, manufacturing, etc.) during refinement
- **Prompt injection protection** — Fast regex-based guards block jailbreak attempts
- **Stateless API** — Full conversation history sent on every request, no server-side sessions

---

## Architecture

```
POST /chat
    │
    ▼
┌──────────────┐
│  guards.py   │  Fast regex guards: injection, legal, off-topic
└──────┬───────┘
       ▼
┌──────────────┐
│  state.py    │  Reconstruct hiring context from message history
└──────┬───────┘
       ▼
┌──────────────┐
│ chat_logic   │  Classify turn → route to handler
└──────┬───────┘
       │
       ├─► clarify     Ask targeted questions → recommendations: []
       ├─► recommend   Retrieve → rank → balance → LLM reply
       ├─► refine      Update shortlist preserving role context
       ├─► compare     Grounded comparison from catalog metadata
       ├─► refuse      Polite refusal for out-of-scope requests
       └─► close       Confirm and lock final shortlist
       │
       ▼
┌──────────────┐
│ formatter.py │  Schema-safe Recommendation objects
└──────┬───────┘
       ▼
ChatResponse { reply, recommendations[], end_of_conversation }
```

### Module Overview

| Module | Role |
|--------|------|
| `agent/guards.py` | Regex-based injection/off-topic detection |
| `agent/state.py` | Stateless context reconstruction from history |
| `agent/retriever.py` | Hybrid FAISS semantic + keyword search |
| `agent/ranker.py` | Multi-signal scoring + battery balancing |
| `agent/recommendation_engine.py` | Query building, shortlist assembly, refinement |
| `agent/comparison.py` | Grounded two-assessment comparison |
| `agent/refusal.py` | Refusal classification and response building |
| `agent/formatter.py` | Schema-safe API response construction |
| `agent/chat_logic.py` | Central orchestration + LLM calls |
| `agent/prompts.py` | All prompt templates |
| `embeddings/build_index.py` | One-time FAISS index builder |
| `models/schemas.py` | Pydantic request/response models |

---

## Quick Start

### Prerequisites

- Python 3.11+
- [OpenRouter API key](https://openrouter.ai/settings/keys) (free tier works)

### 1. Clone & install

```bash
git clone https://github.com/Yashmaggu-NV/shl-assessment-recommender.git
cd shl-assessment-recommender/project
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env → set OPENROUTER_API_KEY=your_key_here
```

### 3. Build the FAISS index

```bash
python embeddings/build_index.py
```

Encodes ~500 SHL catalog items using `all-MiniLM-L6-v2` into a flat FAISS L2 index.

### 4. Start the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The server auto-builds the FAISS index on first startup if not present.

---

## API Reference

### `GET /health`

Readiness probe. Returns `{"status": "ok"}` when the service is ready.

```bash
curl http://localhost:8000/health
```

### `POST /chat`

Stateless conversational endpoint. Every call must include the **full conversation history**.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a senior Java developer with Spring and SQL"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are 6 assessments for a senior Java backend developer.",
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

**Schema rules:**
- `reply` — Always present, natural language response
- `recommendations` — `[]` when clarifying, comparing, or refusing; 1–10 items when recommending
- `end_of_conversation` — `true` only when the user confirms completion

**Multi-turn example:**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a software engineer"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, add personality and teamwork assessments too"}
  ]
}
```

---

## Supported Conversation Patterns

| Pattern | Behavior |
|---------|----------|
| Vague query ("I need an assessment") | Asks clarification, returns `recommendations: []` |
| Role-based ("Hiring a Java developer") | Recommends relevant technical assessments |
| Refinement ("Add personality tests") | Updates shortlist preserving original role context |
| Comparison ("OPQ vs GSA?") | Grounded comparison using catalog data only |
| Prompt injection ("Ignore instructions") | Polite refusal, `recommendations: []` |
| Off-topic ("Should I fire someone?") | Polite refusal, redirects to assessment selection |
| Non-SHL products ("Recommend Coursera") | Refuses, suggests SHL alternatives |

---

## Deployment

### Docker

```bash
cd project
docker build -t shl-recommender .
docker run -p 8000:8000 -e OPENROUTER_API_KEY=your_key shl-recommender
```

### Render

1. Push to GitHub
2. Connect repo to [Render](https://render.com)
3. Set `OPENROUTER_API_KEY` as a secret environment variable
4. Deploy — `render.yaml` handles build and start commands

### Railway

1. Connect GitHub repo to [Railway](https://railway.app)
2. Set `OPENROUTER_API_KEY` in the environment
3. Root directory: `project/`
4. Build: `pip install -r requirements.txt && python embeddings/build_index.py`
5. Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`

---

## Testing

```bash
cd project

# Unit tests
pytest tests/ -v

# Evaluation regression tests (10 scenarios)
python test_evaluation.py

# Quick recommendation test
python test_query.py
```

### Test Coverage

| Suite | What it tests |
|-------|--------------|
| `tests/test_guards.py` | Injection, legal, off-topic detection |
| `tests/test_state.py` | State reconstruction from history |
| `tests/test_retrieval.py` | Retriever, ranker, engine, formatter |
| `test_evaluation.py` | Full evaluation regression (all 10 patterns) |
| `test_query.py` | Java backend + refinement + removal |

---

## Design Decisions

### Statelessness
Every request carries the full conversation history. No databases or sessions. State is reconstructed on each call via LLM extraction + regex fallback.

### Hybrid Retrieval
- **Semantic**: `all-MiniLM-L6-v2` embeddings → FAISS flat L2 (exact search for ~500 items)
- **Keyword**: Jaccard-like token overlap scoring
- **Fusion**: 60% semantic + 40% keyword

### Domain-Aware Filtering (4 layers)
For tech/software roles, unrelated-domain items are blocked at every stage:
1. **Ranker** — `score = 0` for irrelevant domains
2. **Post-rank filter** — Hard removal inside `assemble_recommendations`
3. **Chat logic filter** — `_filter_domain_irrelevant()` after LLM resolution
4. **Formatter** — Last-mile report/guide product filter

### Battery Composition
The ranker composes balanced assessment sets:
- **Tech roles**: K (knowledge) + A (cognitive) + P (personality)
- **Graduate**: A (cognitive) + B (SJT) + P (personality)
- **Contact centre**: S (simulation) + spoken language + P
- **Safety-critical**: P (DSI) + K (safety knowledge)
- **Leadership**: P (OPQ32r) + A (cognitive)

### LLM Fallback Safety
If the LLM fails (rate limit, timeout, error), the system falls back to deterministic catalog retrieval + rule-based responses. No degradation to hallucination.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI + Uvicorn |
| LLM | DeepSeek V4 Flash via OpenRouter |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector Search | FAISS (flat L2, CPU) |
| Schema | Pydantic v2 |
| Deployment | Docker / Render / Railway |

---

## Project Structure

```
project/
├── agent/                  # Core agent modules
│   ├── chat_logic.py       # Central orchestration
│   ├── guards.py           # Prompt injection protection
│   ├── state.py            # State reconstruction
│   ├── retriever.py        # Hybrid search
│   ├── ranker.py           # Scoring & battery balance
│   ├── recommendation_engine.py  # Shortlist assembly
│   ├── comparison.py       # Assessment comparison
│   ├── refusal.py          # Refusal handling
│   ├── formatter.py        # Response formatting
│   └── prompts.py          # Prompt templates
├── data/
│   └── catalog.json        # SHL product catalog (~500 items)
├── embeddings/
│   ├── build_index.py      # FAISS index builder
│   └── faiss_index/        # Pre-built index (generated)
├── models/
│   └── schemas.py          # Pydantic models
├── utils/
│   └── helpers.py          # Shared utilities
├── tests/                  # Unit & integration tests
├── app.py                  # FastAPI application
├── Dockerfile              # Multi-stage Docker build
├── requirements.txt        # Python dependencies
├── render.yaml             # Render deployment config
└── .env.example            # Environment template
```

---

## License

This project was built as part of the SHL GenAI Assessment assignment.
