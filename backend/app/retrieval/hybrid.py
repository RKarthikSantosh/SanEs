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


def hybrid_search(question: str, top_n: int | None = None, rerank: bool = True) -> list[dict]:
    settings = get_settings()
    top_n = top_n or settings.context_k

    # the two retrievers are independent, so run them at the same time
    with ThreadPoolExecutor(max_workers=2) as pool:
        vector = pool.submit(query_vectors, question, settings.retriever_k)
        keyword = pool.submit(query_keywords, question, settings.retriever_k)
        ranked_lists = {"vector": vector.result(), "keyword": keyword.result()}

    fused = reciprocal_rank_fusion(
        ranked_lists,
        k=settings.rrf_k,
        limit=settings.fused_k,
    )
    if not rerank:
        return fused[:top_n]
    return rerank_candidates(question, fused, top_n=top_n)
