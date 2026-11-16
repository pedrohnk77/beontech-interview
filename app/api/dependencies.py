"""Shared providers that wire the services together for the routes."""

from functools import lru_cache

from fastapi import Depends

from app.services import llm
from app.services.guardrails import InjectionGuard
from app.services.knowledge import KnowledgeBase
from app.services.qa import Answerer, AskService, Judge
from app.services.retrieval import RagService, Retriever


@lru_cache
def knowledge_base() -> KnowledgeBase:
    return KnowledgeBase.from_file()


@lru_cache
def retriever() -> Retriever:
    """Built once: the app's lifespan calls this at startup to index the snippets."""
    return Retriever(knowledge_base())


def ask_llm(prompt: str, system: str | None = None) -> str:
    return llm.ask(prompt, system=system)


def ask_service() -> AskService:
    return AskService(Answerer(ask_llm), Judge(ask_llm), InjectionGuard())


def rag_service(service: AskService = Depends(ask_service)) -> RagService:
    return RagService(retriever(), service)
