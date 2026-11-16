import pytest

from app.services import llm
from app.services.qa import GROUNDED


class StubLLM:
    """Stands in for llm.ask and records every system prompt it receives."""

    def __init__(self, draft: str, verdict: str) -> None:
        self.draft = draft
        self.verdict = verdict
        self.systems: list[str] = []

    def __call__(self, prompt: str, system: str | None = None) -> str:
        self.systems.append(system)
        # The judge is the only caller whose system prompt mentions a judge.
        return self.verdict if "judge" in system else self.draft


@pytest.fixture
def stub_llm(monkeypatch):
    def install(draft: str = "A grounded answer.", verdict: str = GROUNDED) -> StubLLM:
        stub = StubLLM(draft, verdict)
        monkeypatch.setattr(llm, "ask", stub)
        return stub

    return install
