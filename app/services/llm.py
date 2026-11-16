"""Thin wrapper around the Anthropic SDK so endpoints stay small."""

import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")

# Reads ANTHROPIC_API_KEY from the environment / .env.
client = anthropic.Anthropic()


class LLMNotConfigured(RuntimeError):
    pass


def ask(prompt: str, system: str | None = None, max_tokens: int = 16000) -> str:
    """Send a single-turn prompt to Claude and return the text reply."""
    if not (client.api_key or client.auth_token):
        raise LLMNotConfigured("ANTHROPIC_API_KEY is not set (copy .env.example to .env)")
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system or anthropic.omit,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": "medium"},
        # Server-side refusal fallback: if the model declines, the API retries
        # on a suitable fallback model inside the same call.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined to answer this request.")
    return "".join(block.text for block in response.content if block.type == "text")
