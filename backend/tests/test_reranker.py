from app.config import get_settings
from app.reranker import rerank_candidates


def test_rerank_candidates_falls_back_to_input_when_reranker_is_unavailable(monkeypatch):
    monkeypatch.setattr(get_settings(), "cohere_api_key", "")
    candidates = [
        {"id": "doc-1", "text": "Alpha chunk", "metadata": {"filename": "a.txt"}},
        {"id": "doc-2", "text": "Beta chunk", "metadata": {"filename": "b.txt"}},
    ]

    reranked = rerank_candidates("What is alpha?", candidates, top_n=1)

    assert len(reranked) == 1
    assert reranked[0]["id"] == "doc-1"
