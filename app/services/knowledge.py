"""The knowledge base: the only source of truth the service may answer from."""

import re
from pathlib import Path

SOT_PATH = Path(__file__).parent.parent / "data" / "sot.txt"

_QUOTED = re.compile(r'"([^"]+)"')


class KnowledgeBase:
    def __init__(self, snippets: list[str]) -> None:
        self.snippets = snippets

    @classmethod
    def from_file(cls, path: Path = SOT_PATH) -> "KnowledgeBase":
        """Each quoted block in the file is one snippet (line wraps are joined)."""
        raw = path.read_text(encoding="utf-8")
        return cls([" ".join(block.split()) for block in _QUOTED.findall(raw)])

    @property
    def text(self) -> str:
        return "\n".join(f"- {snippet}" for snippet in self.snippets)
