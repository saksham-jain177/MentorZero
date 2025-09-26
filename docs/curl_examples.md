# CURL examples

## Health
```bash
curl http://localhost:8000/llm_health
```

## Teach
```bash
curl -X POST http://localhost:8000/teach \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u1","topic":"binary search","mode":"explain"}'
```

## Submit answer
```bash
curl -X POST http://localhost:8000/submit_answer \
  -H "Content-Type: application/json" \
  -d '{"session_id":"sess_u1","interaction_id":"i1","user_answer":"my answer"}'
```

## Upload text
```bash
curl -X POST http://localhost:8000/upload_url_or_text \
  -H "Content-Type: application/json" \
  -d '{"session_id":"sess_u1","url_or_text":"some text to index"}'
```

## Progress
```bash
curl "http://localhost:8000/progress?user_id=u1"
```

