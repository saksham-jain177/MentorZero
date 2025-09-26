# MentorZero

MentorZero is an offline‑first, zero‑cost adaptive tutoring agent. It runs locally with Ollama, performs Agentic RAG for cited answers, and includes AutoLearn (self‑improvement) with a fast LLM‑as‑a‑judge.

## Quick start

Prerequisites:
- Python 3.10+
- Ollama installed and running (http://localhost:11434)

1) Pull a fast local model for responses and judging (recommended):
```
ollama pull llama3.1:8b-instruct-q4_K_M
```

2) Set environment (Windows PowerShell example):
```
$env:MZ_OLLAMA_MODEL = "llama3.1:8b-instruct-q4_K_M"
$env:MZ_JUDGE_MODEL  = "llama3.1:8b-instruct-q4_K_M"
# Optional cascade settings (kept fast by default)
$env:MZ_JUDGE_MARGIN = "0.1"
$env:MZ_JUDGE_TIMEOUT_FAST = "8"
```
If you prefer .env, use the same keys without `$env:`.

3) Run the API:
```
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

4) Open the UI:
- Open `ui/index.html` in your browser (or serve the `ui/` folder statically).

## Agentic RAG
- Upload text directly on the Chat page (Upload is merged into Chat). Your content is chunked, embedded (SentenceTransformers if available), and indexed into FAISS.
- While chatting, answers include top‑k citations from your local index.

## AutoLearn (formerly “AZL”)
- Streams live events (SSE) while generating and judging Q/A examples.
- Judge is fast by default (same `MZ_JUDGE_MODEL`). Heavy judge is disabled unless you set `MZ_JUDGE_MODEL_HEAVY`.
- Config knobs (env with `MZ_` prefix):
  - `MZ_JUDGE_MODEL` – fast judge model (default: your main model)
  - `MZ_JUDGE_MODEL_HEAVY` – optional fallback judge (unset to disable)
  - `MZ_JUDGE_MARGIN` – escalate window around pass threshold (default 0.1)
  - `MZ_JUDGE_TIMEOUT_FAST` – seconds (default 8)
  - `MZ_JUDGE_TIMEOUT_HEAVY` – seconds (default 30)
  - `MZ_AZL_PASS_THRESHOLD`, `MZ_AZL_MAX_ATTEMPTS`, `MZ_AZL_DAILY_BUDGET`, `MZ_AZL_SCORE_WEIGHTS`

## Keyboard shortcuts
- Alt + T: cycle tabs (Home → Chat → About → AutoLearn)
- Alt + L: toggle light/dark theme
- / : focus the Learn input (when Learn is active)

## Troubleshooting
- Ensure Ollama is running and the model is pulled: `ollama list`
- If sentence‑transformers fails to load, embeddings fall back to a deterministic hash (works, but search quality is lower).
- The backend exposes health endpoints: `/llm_health`, `/judge_health`.

## Project layout
- `api/` – FastAPI app & routes (SSE Autolearn, teach, upload)
- `agent/` – services (Agentic RAG, teaching, scoring, validators, config)
- `ui/` – vanilla JS/CSS/HTML frontend

## License
MIT