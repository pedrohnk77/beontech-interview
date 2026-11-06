import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import llm

app = FastAPI(title="BEON.tech Interview")


class EchoRequest(BaseModel):
    message: str


class EchoResponse(BaseModel):
    message: str
    length: int


class ChatRequest(BaseModel):
    prompt: str
    system: str | None = None


class ChatResponse(BaseModel):
    reply: str
    model: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/echo", response_model=EchoResponse)
def echo(body: EchoRequest) -> EchoResponse:
    """Simple POST endpoint: returns what it receives."""
    return EchoResponse(message=body.message, length=len(body.message))


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    """POST endpoint backed by the LLM."""
    try:
        reply = llm.ask(body.prompt, system=body.system)
    except llm.LLMNotConfigured as e:
        raise HTTPException(status_code=500, detail=str(e))
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=500, detail="Invalid or missing ANTHROPIC_API_KEY")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="LLM rate limited, retry shortly")
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e.message}")
    except anthropic.APIConnectionError:
        raise HTTPException(status_code=503, detail="Could not reach the LLM API")
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return ChatResponse(reply=reply, model=llm.MODEL)
