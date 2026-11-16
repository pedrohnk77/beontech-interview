"""HTTP endpoints. They only translate between HTTP and the services."""

from fastapi import APIRouter, Depends

from app.api.dependencies import ask_service, knowledge_base, rag_service
from app.api.errors import LLM_ERRORS, to_http_error
from app.api.schemas import (
    AskResponse,
    ChatRequest,
    ChatResponse,
    EchoRequest,
    EchoResponse,
    QuestionRequest,
    RetrieveResponse,
    Snippet,
)
from app.services import llm
from app.services.qa import AskService
from app.services.retrieval import RagService

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/echo", response_model=EchoResponse)
def echo(body: EchoRequest) -> EchoResponse:
    """Simple POST endpoint: returns what it receives."""
    return EchoResponse(message=body.message, length=len(body.message))


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    """POST endpoint backed by the LLM."""
    try:
        reply = llm.ask(body.prompt, system=body.system)
    except LLM_ERRORS as error:
        raise to_http_error(error) from error
    return ChatResponse(reply=reply, model=llm.MODEL)


@router.post("/ask", response_model=AskResponse)
def ask(body: QuestionRequest, service: AskService = Depends(ask_service)) -> AskResponse:
    """Core task: answers from the whole knowledge base; refuses anything outside it."""
    try:
        reply = service.ask(body.question, knowledge_base())
    except LLM_ERRORS as error:
        raise to_http_error(error) from error
    return AskResponse(question=body.question, answer=reply.answer, refused=reply.refused)


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    body: QuestionRequest, service: RagService = Depends(rag_service)
) -> RetrieveResponse:
    """Bonus task (RAG): answers from the retrieved snippets only."""
    try:
        result = service.ask(body.question)
    except LLM_ERRORS as error:
        raise to_http_error(error) from error
    return RetrieveResponse(
        question=body.question,
        answer=result.reply.answer,
        refused=result.reply.refused,
        snippets=[Snippet(text=hit.text, distance=hit.distance) for hit in result.hits],
    )
