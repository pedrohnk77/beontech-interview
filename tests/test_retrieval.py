"""Retrieval tests run the real Chroma default embeddings; only the LLM is stubbed."""

import pytest
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from fastapi.testclient import TestClient

from app.api import dependencies
from app.main import app
from app.services import retrieval
from app.services.qa import REFUSAL

MISSION = "BEON.tech mission is to place"
SERVICES = "BEON.tech offer IT staff augmentation"
CULTURE = "At BEON.tech, building software means"

client = TestClient(app)


@pytest.fixture(scope="module")
def retriever() -> retrieval.Retriever:
    return dependencies.retriever()


def post(question: str):
    return client.post("/retrieve", json={"question": question})


def test_snippets_are_indexed_with_chroma_default_embeddings(retriever):
    collection = retriever._collection
    assert collection.count() == 3
    assert isinstance(collection._embedding_function, DefaultEmbeddingFunction)


def test_startup_indexes_the_snippets_once():
    dependencies.retriever.cache_clear()
    with TestClient(app):
        assert dependencies.retriever.cache_info().currsize == 1
        indexed = dependencies.retriever()
    assert dependencies.retriever() is indexed


# Paraphrases worded differently from the snippet they should match ("goal" for
# "mission", "hire"/"outsourcing" for "staff augmentation", "develop" for "grow"):
# ranking them correctly relies on meaning (embeddings), not exact wording.
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the company's main goal?", MISSION),
        ("Can I hire a QA engineer through you?", SERVICES),
        ("Do you do outsourcing for frontend work?", SERVICES),
        ("How do you help people develop their careers?", CULTURE),
    ],
)
def test_paraphrased_question_finds_the_right_snippet(retriever, question, expected):
    assert retriever.search(question)[0].text.startswith(expected)


def test_hits_are_limited_to_top_k_and_sorted_nearest_first(retriever):
    hits = retriever.search("Which services does BEON.tech offer?")
    assert len(hits) == retrieval.TOP_K
    assert hits[0].distance <= hits[1].distance


def test_closer_wording_gets_a_smaller_distance(retriever):
    exact = retriever.search("What is BEON.tech's mission?")[0]
    loose = retriever.search("What is the company's main goal?")[0]
    assert exact.distance < loose.distance


@pytest.mark.parametrize(
    "question",
    ["What is the capital of France?", "How do I bake a chocolate cake?", "Tell me a joke"],
)
def test_off_topic_question_retrieves_nothing(retriever, question):
    assert retriever.search(question) == []


def test_endpoint_returns_answer_with_its_snippets(stub_llm):
    stub_llm(draft="IT staff augmentation services.")
    r = post("Which services does BEON.tech offer?")
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "IT staff augmentation services."
    assert body["refused"] is False
    assert body["snippets"][0]["text"].startswith(SERVICES)
    assert len(body["snippets"]) == retrieval.TOP_K


def test_only_retrieved_snippets_reach_the_llm(stub_llm):
    stub = stub_llm()
    post("Which services does BEON.tech offer?")
    assert len(stub.systems) == 2  # answerer + judge
    for system in stub.systems:
        assert SERVICES in system
        assert CULTURE not in system


def test_off_topic_question_is_refused_without_calling_the_llm(stub_llm):
    stub = stub_llm()
    body = post("What is the capital of France?").json()
    assert body == {
        "question": "What is the capital of France?",
        "answer": REFUSAL,
        "refused": True,
        "snippets": [],
    }
    assert stub.systems == []


def test_empty_question_is_rejected():
    assert post("").status_code == 422
