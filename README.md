# BEON.tech AI Engineer Challenge

A small FastAPI service that answers questions about BEON.tech using an LLM (Claude) and a fixed knowledge base of three snippets.

- **Core task**: `POST /ask` sends the whole knowledge base to the LLM and returns a JSON answer.
- **Bonus task (RAG)**: `POST /retrieve` embeds the snippets into ChromaDB, retrieves only the most relevant ones for the question, and answers from those.
- **No hallucinations**: questions outside the knowledge base are refused instead of answered.

## Run it step by step

### 1. Install uv

[uv](https://docs.astral.sh/uv/getting-started/installation/) is the only tool you need. It installs the right Python version (3.12+) for you.

```bash
brew install uv
```

### 2. Get the code and install dependencies

```bash
git clone https://github.com/pedrohnk77/beon-tech.git
cd beon-tech
uv sync
```

### 3. Add your Anthropic API key

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with a real key from https://console.anthropic.com/settings/keys:

```
ANTHROPIC_API_KEY=sk-ant-your-real-key
```

### 4. Start the server

```bash
uv run uvicorn app.main:app --reload
```

Leave this terminal open. The app is ready when you see `Uvicorn running on http://127.0.0.1:8000`.

The first start downloads Chroma's embedding model (about 80 MB, saved to `~/.cache/chroma`), so it can take a minute. Later starts are fast.

### 5. Ask questions

Open a second terminal and try these.

**Core task: answer from the knowledge base**

```bash
curl -X POST localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is BEON.tech mission?"}'
```

You get something like this (the exact wording comes from the LLM):

```json
{
  "question": "What is BEON.tech mission?",
  "answer": "BEON.tech's mission is to place the brightest tech talent in the most disruptive and innovative U.S. companies.",
  "refused": false
}
```

**Bonus task: RAG, the answer comes with the snippets that were retrieved**

```bash
curl -X POST localhost:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"question": "Can I hire a QA engineer through you?"}'
```

The response has the same fields plus `snippets`: the retrieved texts and their distance to the question (lower means closer in meaning).

**A question outside the knowledge base is refused**

```bash
curl -X POST localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the capital of France?"}'
```

You get:

```json
{
  "question": "What is the capital of France?",
  "answer": "I can only answer questions covered by the BEON.tech knowledge base.",
  "refused": true
}
```

Prefer clicking over typing? You have two options:

- Open http://127.0.0.1:8000/docs and try the endpoints from the browser.
- Open `requests.http` in VS Code (REST Client extension) or a JetBrains IDE. It has a ready-made request for every scenario.

To stop the server, press `Ctrl+C` in its terminal.

### 6. Run the tests

```bash
uv run pytest
```

The tests do not need an API key: the LLM is stubbed. The embeddings and ChromaDB are real.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ANTHROPIC_API_KEY is not set` | Do step 3, then restart the server. |
| `Invalid or missing ANTHROPIC_API_KEY` | The key in `.env` is wrong. Paste a valid one and restart the server. |
| `Address already in use` | Port 8000 is taken. Add `--port 8001` to the command in step 4. |
| The first start seems stuck | It is downloading the embedding model (step 4). Wait for it to finish. |

## How it works

```
question
   │
   ▼
InjectionGuard ── prompt injection? ──────────────► refused
   │
   ▼
Retriever (only /retrieve) ── nothing relevant? ──► refused
   │
   ▼
Answerer (LLM) ── not in the knowledge base? ─────► refused
   │
   ▼
Judge (LLM) ── answer not fully supported? ───────► refused
   │
   ▼
answer
```

- **Prompt engineering**: the answerer may use only the knowledge base given in its system prompt, must be concise, and must reply `OUT_OF_SCOPE` when the answer is not there. The question travels inside `<question>` tags and is treated as data, not instructions.
- **LLM-as-a-judge**: a second LLM call checks that every claim in the draft is stated in the knowledge base. Anything other than an exact `GROUNDED` verdict is a refusal.
- **Guardrails**: common prompt-injection patterns are blocked before any LLM call.
- **RAG**: the snippets are embedded with Chroma's default model (all-MiniLM-L6-v2) into an in-memory collection, once, at startup. Each question retrieves the top 2 snippets by cosine distance. Snippets that are too far away are dropped, so off-topic questions are refused without calling the LLM.

## Endpoints

| Endpoint | What it does |
| --- | --- |
| `POST /ask` | Core task. Answers from the whole knowledge base. |
| `POST /retrieve` | Bonus task. Answers from the retrieved snippets only. |
| `GET /health` | Returns `{"status": "ok"}`. |
| `POST /echo` | Returns what it receives. No API key needed. |
| `POST /chat` | Sends a free prompt straight to the LLM. |

## Project layout

```
app/
├── main.py              Creates the FastAPI app, indexes the snippets at startup
├── api/                 HTTP layer
│   ├── routes.py        The endpoints
│   ├── schemas.py       Request and response bodies
│   ├── dependencies.py  Wires the services for the routes
│   └── errors.py        Maps LLM failures to HTTP errors
├── services/            Logic, with no FastAPI in it
│   ├── llm.py           ask(prompt, system) around the Anthropic client
│   ├── knowledge.py     KnowledgeBase, loads the snippets
│   ├── guardrails.py    InjectionGuard
│   ├── qa.py            Answerer, Judge, AskService and their prompts
│   └── retrieval.py     Retriever (ChromaDB) and RagService
└── data/
    └── sot.txt          The knowledge base: the three snippets
tests/                   pytest + FastAPI TestClient
requests.http            Ready-made requests for every scenario
```

## Changing the model

The default model is `claude-haiku-5-5`. To use another one, add this line to `.env` and restart the server:

```
ANTHROPIC_MODEL=claude-sonnet-5-5
```
