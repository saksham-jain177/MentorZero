# MentorZero

Local, private, adaptive tutoring agent with Agentic RAG and self‑improvement inspired by “Absolute Zero: Reinforced Self‑play Reasoning with Zero Data.” The system runs entirely on your machine via Ollama, streams progress in real time, and requires no cloud services.

Terminology
- We use the term “AutoLearn” in the UI. Conceptually it maps to “Absolute Zero” style self‑play with zero external labels: the agent generates data, evaluates it, and accepts only high‑quality items for later use. Our current implementation optimizes prompts and retrieval rather than updating base model weights. A PEFT track (LoRA/QLoRA) is optional and can be added later to adapt weights using the accepted examples.

---

## Absolute Zero (AZ): Reinforced Self‑play Reasoning with Zero Data

Absolute Zero improves reasoning without labeled data via self‑play and self‑evaluation with reinforcement‑style acceptance. The agent proposes candidates, scores them under internal rules and an LLM‑as‑judge, and keeps only the highest‑quality outputs.

What we ship now
- Self‑generation of Q/A examples by topic
- Multi‑check validators + LLM‑as‑judge scoring with fast→heavy judge cascade
- Auto‑acceptance using a configurable threshold and judge margin
- Cached evaluations (RAM + SQLite) to keep self‑play affordable locally
- Reuse of accepted examples as few‑shots at inference

Not default (by design)
- No iterative weight updates. We optimize retrieval, prompting, and caching for local UX.

Optional PEFT (opt‑in)
- Export accepted examples → LoRA/QLoRA → adapter weights → serve via an Ollama‑compatible runtime

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

## Metrics to track

- Acceptance rate: accepted / generated
- Judge escalation rate: fraction escalated to heavy judge
- Cache hit rate: evaluations served from cache
- Latency p50/p95 (ms): generation; judge (fast/heavy)
- Average attempts per accepted example
- Failure rate: validation or judge errors
- Throughput: examples accepted per minute

Observability
- GET `/metrics/performance` for aggregate timings and cache stats
- JSONL request log at `data/ollama_requests.log` with timestamp, duration_ms, model, endpoint, status, prompt_chars, temperature, meta

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