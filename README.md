# MentorZero

Local, private, adaptive tutoring agent with Agentic RAG and self‑improvement inspired by “Absolute Zero: Reinforced Self‑play Reasoning with Zero Data.” The system runs entirely on your machine via Ollama, streams progress in real time, and requires no cloud services.

Terminology
- We use the term “AutoLearn” in the UI. Conceptually it maps to “Absolute Zero” style self‑play with zero external labels: the agent generates data, evaluates it, and accepts only high‑quality items for later use. Our current implementation optimizes prompts and retrieval rather than updating base model weights. A PEFT track (LoRA/QLoRA) is optional and can be added later to adapt weights using the accepted examples.

---

## Deliverables (what’s included)
- Backend API (FastAPI) with:
  - Teach/chat endpoints with retrieval‑augmented generation (hybrid dense+BM25; MMR diversification; optional query rewriting)
  - AutoLearn streaming endpoint (Server‑Sent Events) with judge cascade and cache
  - Upload/ingest endpoints (text/url) with chunking, embedding, FAISS indexing
  - Health endpoints (`/llm_health`, `/judge_health`), metrics (`/metrics/performance`)
  - Background task processor for long‑running jobs
- Agent services:
  - Optimized AutoLearn scorer: in‑memory + SQLite cache; fast→heavy judge cascade; error instrumentation
  - Embedding service (SentenceTransformers if available; deterministic fallback when offline)
  - Vector store (FAISS local index)
  - Teaching orchestration (RAG + few‑shot with accepted examples)
- Data layer:
  - SQLite with SQLAlchemy models for Users/Sessions/Interactions
  - Synthetic examples store
  - EvaluationCache and PerformanceMetrics tables
- Frontend (vanilla JS/CSS/HTML):
  - Minimalist Chat + integrated Upload (collapsible) + AutoLearn Observer
  - Live SSE observer with granular events, copy/download log, progress bar, and judge rationales
  - Local persistence (chat history, optimization versions, theme, help seen)
  - Global Help guide (floating “?”) with consistent modal UX across pages
- Dev/test assets:
  - End‑to‑end test script, sample integration tests
  - Automated table creation on startup; standalone `migrate_db.py` kept for manual runs

---

## Absolute Zero (AZ) vs our AutoLearn

Absolute Zero (AZ) refers to a family of methods that improve reasoning without labeled data by self‑play/self‑evaluation and reinforcement‑style acceptance. In practice:
- The agent proposes tasks/answers, evaluates quality with internal rules/LLM‑as‑judge, and reinforces by keeping the best outputs.

What we implement now
- Self‑generation of Q/A examples by topic
- Multi‑check validation + LLM‑as‑a‑judge scoring
- Auto‑acceptance with a configurable threshold and judge cascade
- Caching of evaluations (memory + SQLite) to make self‑play affordable locally
- Reuse of accepted examples as few‑shots at inference (prompt‑level “learning”)

What we intentionally do not do by default
- We do not update base model weights every iteration. For local UX we focus on retrieval, prompting, and caching. A PEFT fine‑tuning step can be added to adapt weights to your accepted data when desired.

Optional PEFT track (planned/opt‑in)
- Export accepted examples → LoRA/QLoRA → adapter weights → serve via Ollama or another runtime that supports adapters.

---

## Architecture (high‑level)
1) Chat request → Query classifier → (optional) query rewrite → Hybrid RAG (FAISS dense + BM25) with MMR → Compose prompt with citations + few‑shot from accepted examples → LLM generation.
2) AutoLearn loop (SSE): generate candidates → run validators → fast judge; if borderline/low‑confidence → heavy judge → accept or regenerate; stream every step to UI.
3) Caching & resilience: evaluation cache (RAM + SQLite), exponential backoff for LLM/embeddings, circuit‑like early exits on persistent failures.

---

## Quick start

Prerequisites
- Python 3.10+
- Ollama running locally (`http://localhost:11434`)

1) Pull a local model (responses and fast judge).
```
ollama pull llama3.1:8b-instruct-q4_K_M
```

2) Configure environment (PowerShell examples).
```
$env:MZ_OLLAMA_MODEL = "llama3.1:8b-instruct-q4_K_M"
$env:MZ_JUDGE_MODEL  = "llama3.1:8b-instruct-q4_K_M"
$env:MZ_JUDGE_MARGIN = "0.10"      # threshold band to escalate to heavy judge
$env:MZ_JUDGE_TIMEOUT_FAST = "8"   # seconds
# Optional heavy judge (slower reasoning model)
# $env:MZ_JUDGE_MODEL_HEAVY = "gpt-oss:20b"
# $env:MZ_JUDGE_TIMEOUT_HEAVY = "30"

# RAG defaults (override as desired)
$env:MZ_RAG_TOP_K = "5"
$env:MZ_RAG_LAMBDA_MULT = "0.65"
$env:MZ_RAG_HYBRID_ALPHA = "0.5"
```

3) Run the API.
```
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

4) Open the UI.
- Navigate to `http://localhost:8000/ui/index.html` (static files served by the API).

---

## Using MentorZero

Chat (main workflow)
- Ask a topic or a question. The system retrieves your uploaded context and reuses accepted examples as few‑shots. Answers stream and can be optimized.

Upload knowledge (RAG)
- On the Chat page, open “Upload Knowledge,” paste text or fetch from URL, and index it locally. Indexed chunks appear as citations in future answers.

AutoLearn (Absolute‑Zero‑style self‑play)
- Enter a topic and start. Watch the SSE observer show proposal → validation checks → judge decisions → accept/regenerate. Accepted examples are stored and reused automatically.

What to do after AutoLearn
- Start chatting on the same topic to benefit from improved few‑shots.
- Download the AutoLearn log for review or export accepted examples for later PEFT.

Evaluating changes
- Use `test_end_to_end.py` to exercise Optimized AutoLearn, RAG, error handling, and background processing. Adjust constants at the top (topic, count, text) and re‑run to see how scores, cache hits, and timings change. The test prints cache statistics and operation timings so you can compare runs.

---

## Configuration (selected)
- `MZ_OLLAMA_MODEL` – model for generation
- `MZ_JUDGE_MODEL` – fast judge model (default = generation model)
- `MZ_JUDGE_MODEL_HEAVY` – optional slower reasoning judge
- `MZ_JUDGE_MARGIN` – score margin around threshold that triggers heavy judge
- `MZ_AZL_PASS_THRESHOLD` – acceptance threshold (default 0.75)
- `MZ_AZL_MAX_ATTEMPTS` – max regenerations per item
- `MZ_AZL_DAILY_BUDGET` – per‑day example budget
- `MZ_AZL_SCORE_WEIGHTS` – JSON weights for validators and judge
- `MZ_RAG_TOP_K`, `MZ_RAG_LAMBDA_MULT`, `MZ_RAG_HYBRID_ALPHA` – retrieval controls

Automations
- Database tables auto‑create on app startup; manual `migrate_db.py` is available but not required each run.

---

## Endpoints (high‑level)
- `POST /teach` – start chat session
- `POST /submit` – submit answer or follow‑up
- `POST /upload` – ingest text
- `POST /scrape` – fetch URL and ingest
- `GET  /azl/autolearn_stream` – real‑time AutoLearn (SSE)
- `POST /azl/propose` / `POST /azl/validate` / `POST /azl/accept` / `POST /azl/regenerate`
- `GET  /metrics/performance` – aggregate timings and cache stats
- `GET  /llm_health`, `GET /judge_health` – model health

---

## Performance & reliability
- Judge cascade: fast judge for most cases; escalate to heavy judge only when within `MZ_JUDGE_MARGIN` or low‑confidence
- Caching: in‑memory + SQLite for judge/evaluations to avoid recomputation
- Robust parsing of model JSON; retries with backoff; informative SSE errors and suggestions
- Background processor for long tasks (non‑blocking UI)

Request logging
- Every Ollama call is written to `data/ollama_requests.log` (JSONL): timestamp, duration, model, endpoint, status (success/error), prompt size, temperature, and raw response metadata when available. Use it to compute latency distributions and failure rates, or to compare configurations.

Tuning tips
- For slow GPUs/CPUs, keep only a fast judge and raise `MZ_JUDGE_MARGIN` slightly to reduce heavy calls.
- Reduce `MZ_RAG_TOP_K` to lower latency on retrieval‑heavy queries.

---

## Security & privacy
- Runs locally; no cloud calls required.
- Data lives in `data/mentorzero.db` and FAISS index in `data/`.
- You control what is indexed and can delete local artifacts at any time.

---

## Project layout
- `api/` – FastAPI app & routes (teach, upload, SSE AutoLearn, metrics, health)
- `agent/` – services (RAG, teaching, optimized scorer, validators, config)
- `ui/` – vanilla JS/CSS/HTML frontend
- `data/` – SQLite DB and FAISS index (generated at runtime)

---

## License
MIT