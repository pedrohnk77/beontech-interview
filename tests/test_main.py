from fastapi.testclient import TestClient

from app.services import llm
from app.main import app

client = TestClient(app)


def test_echo():
    r = client.post("/echo", json={"message": "hello"})
    assert r.status_code == 200
    assert r.json() == {"message": "hello", "length": 5}


def test_echo_validation():
    r = client.post("/echo", json={})
    assert r.status_code == 422


def test_chat_uses_llm(monkeypatch):
    monkeypatch.setattr(llm, "ask", lambda prompt, system=None: f"stub: {prompt}")
    r = client.post("/chat", json={"prompt": "hi"})
    assert r.status_code == 200
    assert r.json()["reply"] == "stub: hi"
