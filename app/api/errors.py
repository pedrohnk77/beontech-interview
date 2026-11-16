"""Maps LLM failures to HTTP errors so endpoints stay small."""

import anthropic
from fastapi import HTTPException

from app.services import llm

LLM_ERRORS = (anthropic.AnthropicError, RuntimeError)


def to_http_error(error: Exception) -> HTTPException:
    match error:
        case llm.LLMNotConfigured():
            return HTTPException(status_code=500, detail=str(error))
        case anthropic.AuthenticationError():
            return HTTPException(status_code=500, detail="Invalid or missing ANTHROPIC_API_KEY")
        case anthropic.RateLimitError():
            return HTTPException(status_code=429, detail="LLM rate limited, retry shortly")
        case anthropic.APIStatusError():
            return HTTPException(status_code=502, detail=f"LLM error: {error.message}")
        case anthropic.APIConnectionError():
            return HTTPException(status_code=503, detail="Could not reach the LLM API")
        case _:
            return HTTPException(status_code=422, detail=str(error))
