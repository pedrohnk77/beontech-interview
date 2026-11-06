# beon-tech-interview

Scaffold for the BEON.tech AI-focused live coding interview: a FastAPI app with a simple POST endpoint and an LLM (Claude via the Anthropic SDK) ready to use.

## Setup

```bash
uv sync
cp .env.example .env   # then set ANTHROPIC_API_KEY
```

## Run

```bash
uv run uvicorn app.main:app --reload
```

Interactive docs: http://127.0.0.1:8000/docs

```bash
# Simple POST endpoint
curl -X POST localhost:8000/echo -H 'Content-Type: application/json' -d '{"message": "hello"}'

# LLM-backed POST endpoint
curl -X POST localhost:8000/chat -H 'Content-Type: application/json' -d '{"prompt": "Say hi in one sentence"}'
```

## Test

```bash
uv run pytest
```

## Layout

- `app/main.py`: FastAPI app (`/health`, `POST /echo`, `POST /chat`)
- `app/llm.py`: `ask(prompt, system=None)` helper around the Anthropic client
- `tests/`: pytest + FastAPI `TestClient` (the LLM is stubbed)
