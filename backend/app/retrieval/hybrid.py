from concurrent.futures import ThreadPoolExecutor

from app.config import get_settings
from app.reranker import rerank_candidates
from app.retrieval.bm25_store import query_keywords
from app.retrieval.chroma_store import query_vectors


def reciprocal_rank_fusion(ranked_lists: dict[str, list[dict]], k: int = 60, limit: int = 20) -> list[dict]:
    """Merge ranked candidate lists: score(d) = sum over lists of 1 / (k + rank of d)."""
    merged: dict[str, dict] = {}

    for source, candidates in ranked_lists.items():
        for rank, candidate in enumerate(candidates, start=1):
            candidate_id = candidate["id"]
            if candidate_id not in merged:
                merged[candidate_id] = {
                    "id": candidate_id,
                    "text": candidate.get("text"),
                    "metadata": candidate.get("metadata"),
                    "score": 0.0,
                    "sources": [],
                }
            merged[candidate_id]["score"] += 1.0 / (k + rank)
            merged[candidate_id]["sources"].append(source)

    return sorted(merged.values(), key=lambda item: item["score"], reverse=True)[:limit]


def _resolve_explicit_k(top_n: int | str | None) -> int | None:
    if isinstance(top_n, str):
        val = top_n.strip().lower()
        if val in ("auto", "none", "", "dynamic"):
            return None
        if val.isdigit():
            return max(1, min(int(val), 20))
    elif isinstance(top_n, (int, float)) and top_n > 0:
        return max(1, min(int(top_n), 20))
    return None


def hybrid_search(question: str, top_n: int | str | None = None, rerank: bool = True) -> list[dict]:
    settings = get_settings()
    explicit_k = _resolve_explicit_k(top_n)

    # Candidate pool from initial retrievers
    pool_size = max(explicit_k or settings.context_k, settings.retriever_k)
    rerank_target = explicit_k or settings.fused_k

    # the two retrievers are independent, so run them at the same time
    with ThreadPoolExecutor(max_workers=2) as pool:
        vector = pool.submit(query_vectors, question, pool_size)
        keyword = pool.submit(query_keywords, question, pool_size)
        ranked_lists = {"vector": vector.result(), "keyword": keyword.result()}

    fused = reciprocal_rank_fusion(
        ranked_lists,
        k=settings.rrf_k,
        limit=settings.fused_k,
    )

    if not rerank:
        return fused[:(explicit_k or settings.context_k)]

    reranked = rerank_candidates(question, fused, top_n=rerank_target)
    if not reranked:
        return []

    # If explicit top_n requested, return exactly that many
    if explicit_k is not None:
        return reranked[:explicit_k]

    # Dynamic top_n: filter based on relevance score
    best_score = reranked[0].get("rerank_score")
    if best_score is not None and best_score > 0:
        threshold = max(0.1, best_score * 0.4)
        filtered = [c for c in reranked if c.get("rerank_score", 0.0) >= threshold]
        return filtered[:8] if len(filtered) >= 2 else reranked[:min(3, len(reranked))]

    return reranked[:settings.context_k]
