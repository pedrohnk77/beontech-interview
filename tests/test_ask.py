from fastapi.testclient import TestClient

from app.services.knowledge import KnowledgeBase
from app.main import app
from app.services.qa import OUT_OF_SCOPE, REFUSAL

client = TestClient(app)


def ask(question: str):
    return client.post("/ask", json={"question": question})


def test_knowledge_base_loads_three_snippets():
    snippets = KnowledgeBase.from_file().snippets
    assert len(snippets) == 3
    assert snippets[0].startswith("BEON.tech mission is to place")
    assert "\n" not in snippets[0]


def test_grounded_answer_is_returned(stub_llm):
    stub_llm(draft="To place the brightest tech talent.")
    r = ask("What is BEON.tech's mission?")
    assert r.status_code == 200
    assert r.json() == {
        "question": "What is BEON.tech's mission?",
        "answer": "To place the brightest tech talent.",
        "refused": False,
    }


def test_out_of_scope_question_is_refused(stub_llm):
    stub_llm(draft=OUT_OF_SCOPE)
    body = ask("What is the capital of France?").json()
    assert body["refused"] is True
    assert body["answer"] == REFUSAL


def test_answer_rejected_by_judge_is_refused(stub_llm):
    stub_llm(draft="BEON.tech was founded in 1999.", verdict="UNGROUNDED")
    body = ask("When was BEON.tech founded?").json()
    assert body["refused"] is True
    assert body["answer"] == REFUSAL


def test_unexpected_judge_verdict_fails_closed(stub_llm):
    stub_llm(draft="Something.", verdict="Sure, it looks fine!")
    assert ask("Anything?").json()["refused"] is True


def test_empty_question_is_rejected():
    assert ask("").status_code == 422
