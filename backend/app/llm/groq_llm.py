import time

from langchain_groq import ChatGroq

from app.config import get_settings


def _api_keys() -> list[str]:
    settings = get_settings()
    keys = [key for key in (settings.groq_api_key, settings.groq_api_key_fallback) if key]
    if not keys:
        raise ValueError("GROQ_API_KEY is not configured")
    return keys


def get_llm(api_key: str | None = None, model: str | None = None) -> ChatGroq:
    settings = get_settings()
    return ChatGroq(
        model=model or settings.groq_model,
        temperature=0,
        api_key=api_key or _api_keys()[0],
        max_retries=0,
        reasoning_effort=settings.groq_reasoning_effort,
    )


def _is_retryable(error: Exception) -> bool:
    """Rate limits, dropped connections, timeouts and server errors are worth retrying."""
    name, text = type(error).__name__, str(error)
    return (
        any(kind in name for kind in ("RateLimit", "APIConnection", "APITimeout", "InternalServer", "ServiceUnavailable"))
        or any(code in text for code in ("429", "500", "502", "503", "504"))
        or "rate_limit" in text.lower()
    )


# Calls Groq with key rotation
def invoke_with_fallback(messages, model: str | None = None, max_rounds: int = 6) -> str:
    keys = _api_keys()
    delay = 5.0
    last_error: Exception | None = None

    for _ in range(max_rounds):
        for key in keys:
            try:
                return get_llm(api_key=key, model=model).invoke(messages).content
            except Exception as error:
                if not _is_retryable(error):
                    raise
                last_error = error
        time.sleep(delay)
        delay = min(delay * 2, 60.0)

    raise RuntimeError(f"Groq still failing after {max_rounds} rounds") from last_error
