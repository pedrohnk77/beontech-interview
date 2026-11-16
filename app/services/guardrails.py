"""Input guardrails: block prompt-injection attempts before they reach the LLM.

This is the first of three layers. The prompts also treat the question as data,
and the judge rejects any answer that is not grounded in the knowledge base.
"""

import re

INJECTION_PATTERNS = (
    # "ignore all previous instructions", "disregard your rules", ...
    r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(instructions?|prompts?|rules|guidelines)\b",
    # "reveal your system prompt", "print the instructions", ...
    r"\b(reveal|show|print|repeat|leak|output)\b.{0,40}\b(prompts?|instructions?)\b",
    r"\bsystem (prompt|message)\b",
    r"\bdeveloper (mode|message)\b",
    r"\bnew instructions?\b",
    r"\bfrom now on\b",
    r"\byou are now\b",
    r"\bpretend (to be|you)\b",
    r"\bjailbreak\b",
    # Markup that tries to break out of the prompt's data tags.
    r"<\s*/?\s*(question|answer|knowledge_base|system|instructions?)\b",
)


class InjectionGuard:
    def __init__(self, patterns: tuple[str, ...] = INJECTION_PATTERNS) -> None:
        self._patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def blocks(self, text: str) -> bool:
        flat = " ".join(text.split())
        return any(pattern.search(flat) for pattern in self._patterns)
