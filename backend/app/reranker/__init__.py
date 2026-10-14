from typing import Any

from app.config import get_settings

try:
    import cohere
except ImportError:  # pragma: no cover - optional dependency
    cohere = None


# Reranks candidates with Cohere if available; falls back to original order otherwise.
def rerank_candidates(question: str, candidates: list[dict[str, Any]], top_n: int = 5) -> list[dict[str, Any]]:
    if not candidates:
        return []

    if not question:
        return candidates[:top_n]

    settings = get_settings()
    api_key = settings.cohere_api_key
    if not api_key or cohere is None:
        return candidates[:top_n]

    try:
        client = cohere.Client(api_key)
        response = client.rerank(
            query=question,
            documents=[candidate.get("text") or "" for candidate in candidates],
            top_n=min(top_n, len(candidates)),
            model=settings.cohere_rerank_model,
        )
    except Exception:
        return candidates[:top_n]

    reranked: list[dict[str, Any]] = []
    results = getattr(response, "results", None) or []

    for item in results:
        if isinstance(item, dict):
            index = item.get("index")
            score = item.get("relevance_score")
        else:
            index = getattr(item, "index", None)
            score = getattr(item, "relevance_score", None)

        if index is None or index >= len(candidates):
            continue

        candidate = dict(candidates[index])
        candidate["rerank_score"] = float(score or 0.0)
        candidate["score"] = candidate.get("score", 0.0)
        reranked.append(candidate)

    if reranked:
        return reranked

    return candidates[:top_n]
