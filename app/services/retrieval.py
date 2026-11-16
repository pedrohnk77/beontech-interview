"""RAG: retrieve the most relevant snippets with embeddings, then answer from them.

Snippets are embedded with Chroma's default embedding function and stored in an
in-memory collection. Only the retrieved snippets are passed to the LLM.
"""

from dataclasses import dataclass

import chromadb

from app.services.knowledge import KnowledgeBase
from app.services.qa import AskService, Reply

COLLECTION = "knowledge"
TOP_K = 2
# Cosine distance (0 = same meaning). Off-topic questions score above this.
MAX_DISTANCE = 0.9


@dataclass(frozen=True)
class Hit:
    text: str
    distance: float


@dataclass(frozen=True)
class RagReply:
    reply: Reply
    hits: list[Hit]


class Retriever:
    """Semantic search over the knowledge base snippets."""

    def __init__(
        self, knowledge: KnowledgeBase, top_k: int = TOP_K, max_distance: float = MAX_DISTANCE
    ) -> None:
        self._top_k = min(top_k, len(knowledge.snippets))
        self._max_distance = max_distance
        self._collection = self._index(knowledge.snippets)

    @staticmethod
    def _index(snippets: list[str]) -> chromadb.Collection:
        client = chromadb.EphemeralClient()
        collection = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        collection.upsert(ids=[str(i) for i in range(len(snippets))], documents=snippets)
        return collection

    def search(self, question: str) -> list[Hit]:
        """Return the closest snippets, nearest first, dropping irrelevant ones."""
        found = self._collection.query(query_texts=[question], n_results=self._top_k)
        hits = map(Hit, found["documents"][0], found["distances"][0])
        return [hit for hit in hits if hit.distance <= self._max_distance]


class RagService:
    """Answers a question using only the snippets retrieved for it."""

    def __init__(self, retriever: Retriever, ask_service: AskService) -> None:
        self._retriever = retriever
        self._ask_service = ask_service

    def ask(self, question: str) -> RagReply:
        hits = self._retriever.search(question)
        relevant = KnowledgeBase([hit.text for hit in hits])
        return RagReply(reply=self._ask_service.ask(question, relevant), hits=hits)
