# SHL Assessment Recommender

A production-grade conversational agent that guides hiring managers from a vague intent ("I'm hiring a Java developer") to a grounded SHL assessment shortlist through multi-turn dialogue.

## Architecture Overview

```
POST /chat
    │
    ▼
[guards.py]          Fast regex guards: injection, legal, off-topic
    │
    ▼
[state.py]           Reconstruct hiring context from full message history (stateless)
    │
    ▼
[chat_logic.py]      Classify turn: clarify | recommend | refine | compare | refuse | close
    │
    ├─► clarify      Ask 1-2 targeted questions → recommendations: []
    ├─► recommend    Hybrid retrieve → rank → compose battery → LLM reply
    ├─► refine       Apply add/remove/replace to current shortlist
    ├─► compare      Ground comparison from catalog metadata → LLM reply
    ├─► refuse       Polite refusal, stay in scope
    └─► close        Repeat final shortlist, end_of_conversation: true
    │
    ▼
[formatter.py]       Catalog-grounded Recommendation objects (no hallucinated URLs)
    │
    ▼
ChatResponse { reply, recommendations, end_of_conversation }
```

### Key Modules

| Module | Responsibility |
|--------|---------------|
| `agent/guards.py` | Regex-based injection/off-topic detection (no LLM) |
| `agent/state.py` | Stateless context reconstruction from message history |
| `agent/retriever.py` | Hybrid FAISS semantic + keyword search with metadata filtering |
| `agent/ranker.py` | Multi-signal scoring: relevance + job-level + battery balance |
| `agent/recommendation_engine.py` | Query building, shortlist assembly, refinement ops |
| `agent/comparison.py` | Grounded two-assessment comparison using catalog metadata |
| `agent/refusal.py` | Refusal classification and response building |
| `agent/formatter.py` | Schema-safe API response construction |
| `agent/chat_logic.py` | Central orchestration, LLM calls, turn routing |
| `embeddings/build_index.py` | One-time FAISS index builder |

---

## Setup

### Prerequisites
- Python 3.11+
- A free [Gemini API key](https://aistudio.google.com/app/apikey)

### 1. Install dependencies

```bash
cd project
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here
```

### 3. Build the FAISS index

```bash
python embeddings/build_index.py
```

This encodes all ~500 SHL catalog assessments with `all-MiniLM-L6-v2` and saves the index to `embeddings/faiss_index/`.

### 4. Start the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The server auto-builds the FAISS index on first startup if not present.

---

## API Usage

### GET /health

```bash
curl http://localhost:8000/health
# → {"status": "ok"}
```

### POST /chat

Every call carries the **full conversation history**. The service is fully stateless.

**Single turn:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hiring a senior Java developer with Spring and SQL"}
    ]
  }'
```

**Multi-turn (full history):**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hiring a Java developer"},
      {"role": "assistant", "content": "What seniority level?"},
      {"role": "user", "content": "Senior, 7+ years, backend Spring and SQL"}
    ]
  }'
```

**Response schema (non-negotiable):**
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

- `recommendations` is `[]` when clarifying, comparing, or refusing.
- `recommendations` has 1–10 items when recommending or refining.
- `end_of_conversation` is `true` only when the user confirms they're done.

---

## Deployment

### Docker

```bash
# Build (index is baked in at build time)
docker build -t shl-recommender .

# Run
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key shl-recommender
```

### Render

1. Fork / push this repo to GitHub
2. Connect repo to [Render](https://render.com)
3. Set `GEMINI_API_KEY` as a secret environment variable in the Render dashboard
4. Deploy — `render.yaml` handles build and start commands automatically

Build command: `pip install -r requirements.txt && python embeddings/build_index.py`  
Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1`

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Individual suites
pytest tests/test_guards.py -v     # Guard/refusal unit tests (no LLM needed)
pytest tests/test_state.py -v      # State reconstruction unit tests
pytest tests/test_retrieval.py -v  # Retriever, ranker, engine, formatter
pytest tests/test_api.py -v        # Full API integration tests (needs GEMINI_API_KEY)
```

---

## Design Decisions

### Statelessness
Every `POST /chat` call receives the full conversation history. No server-side sessions, databases, or memory are used. State is reconstructed on every call from the message history using both LLM extraction (for nuance) and regex parsing (for speed and reliability as fallback).

### Retrieval Strategy: Hybrid
- **Semantic search**: `all-MiniLM-L6-v2` embeddings in a flat FAISS L2 index. Fast, exact-search, no approximation errors for a ~500-item catalog.
- **Keyword overlap**: Jaccard-like overlap scoring between query tokens and catalog item text.
- **Score fusion**: 60% semantic + 40% keyword, ensuring both signals contribute.
- **Metadata filtering**: Hard filters for excluded categories/names; soft boosts for job-level and language matches.

### Recommendation Composition
The ranker goes beyond pure similarity to compose **balanced assessment batteries**:
- Technical roles: K (knowledge) + A (cognitive) + P (personality)
- Graduate cohorts: A (cognitive) + B (SJT) + P (personality)
- Contact centre: S (simulation) + spoken language + P (personality)
- Safety-critical: P (DSI/Safety instruments) + K (safety knowledge)
- Leadership: P (OPQ32r) + A (cognitive) + report products

### Hallucination Prevention
1. All URLs come exclusively from catalog `link` fields — never generated by the LLM.
2. LLM-returned assessment names are resolved back to catalog items before returning. Unmatched names are dropped with a warning.
3. Comparison answers are built from catalog metadata only, injected into a strict prompt.

### Prompt Injection Protection
Fast regex pattern matching (`guards.py`) runs before any LLM call. Patterns cover: ignore-instructions, role-override, system-prompt-leak, jailbreak, DAN-mode, and similar attacks.

### Turn Classification (No LLM)
Turn type (clarify / recommend / refine / compare / refuse / close) is classified deterministically before the LLM is called, keeping latency low for common patterns.

---

## Evaluation Strategy

### Optimised for Recall@10
- Retrieval pool is 40 candidates before ranking, ensuring relevant items aren't dropped early.
- Battery balance heuristic adds underrepresented categories rather than returning 10 identical-type items.
- Explicitly included items (from history) are injected into the candidate pool and always ranked first.

### Behavior Probes
- **Vague query → clarify**: `is_vague_request()` gates recommendation on turn 1.
- **No hallucination**: URL grounding check in `formatter.py`; LLM name resolution in `chat_logic.py`.
- **Refinement honored**: `detect_refinement_intent()` + `apply_refinement()` apply precise add/remove/replace.
- **Comparison grounded**: `comparison.py` uses only catalog metadata; LLM prompt is strictly scoped.
- **Off-topic refused**: Guards + `classify_refusal()` cover legal, injection, competitor, and general off-topic.

### Turn Cap (Max 8)
The agent recommends after 1–2 clarification turns. If context is present in turn 1 (e.g., a job description), it recommends immediately.

---

## Limitations and Tradeoffs

| Limitation | Tradeoff Made |
|------------|--------------|
| FAISS flat-L2 index | Exact search, no ANN approximation — fine for ~500 items |
| `all-MiniLM-L6-v2` model | Fast (80ms/batch), 384-dim, good English performance — not multilingual |
| Gemini 2.0 Flash | Low latency, cost-effective; lower quality than Pro for complex comparisons |
| No session storage | Full statelessness per spec; adds ~10ms overhead for state reconstruction |
| Regex-based guards | May miss novel injection patterns; LLM-based guard would be more robust but slower |
| Keyword fallback | If FAISS index not built, keyword-only search has lower recall |

---

## Test API Keys

Set `GEMINI_API_KEY` in your environment. Free-tier Gemini keys work for evaluation.  
If no key is set, the system falls back to deterministic retrieval + rule-based responses (no LLM). Most behavior probes will still pass; Recall@10 may be reduced.
