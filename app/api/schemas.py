"""Request and response bodies for the HTTP endpoints."""

from pydantic import BaseModel, Field


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


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool


class Snippet(BaseModel):
    text: str
    distance: float


class RetrieveResponse(AskResponse):
    snippets: list[Snippet]
