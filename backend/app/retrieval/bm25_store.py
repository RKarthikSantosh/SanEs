import re

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from app.retrieval.chroma_store import get_collection

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# cached index over the Chroma collection; rebuilt only after new uploads
_index: dict = {"bm25": None, "ids": [], "documents": [], "metadatas": []}


def tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN_PATTERN.findall(text.lower()) if token not in ENGLISH_STOP_WORDS]


class BM25Index:
    """BM25 over an in-memory list of documents (shared by the API and the experiments)."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.bm25 = BM25Okapi([tokenize(document) or ["_"] for document in documents], k1=k1, b=b)

    def search(self, query: str, n_results: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:n_results]
        return [(index, float(scores[index])) for index in ranked]


def invalidate_index() -> None:
    _index["bm25"] = None


def _load_index() -> None:
    results = get_collection().get(include=["documents", "metadatas"])
    _index["ids"] = results.get("ids", [])
    _index["documents"] = results.get("documents", [])
    _index["metadatas"] = results.get("metadatas", [])
    _index["bm25"] = BM25Index(_index["documents"]) if _index["documents"] else None


def query_keywords(question: str, n_results: int = 5) -> list[dict]:
    if _index["bm25"] is None:
        _load_index()
    if _index["bm25"] is None:
        return []

    return [
        {
            "id": _index["ids"][index],
            "text": _index["documents"][index],
            "metadata": _index["metadatas"][index],
            "score": score,
        }
        for index, score in _index["bm25"].search(question, n_results)
    ]
